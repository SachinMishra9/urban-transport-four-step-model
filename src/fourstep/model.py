"""Four-step model with feedback between assignment and demand.

Congested car times from assignment are fed back into mode choice and
distribution and averaged with the method of successive averages (MSA),
so demand and network speeds end up consistent with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import Network
from .steps import calibrate_beta, frank_wolfe, gravity_model, mode_choice, trip_generation

B_TIME = 0.05        # utility per in-vehicle minute
B_COST = 0.012       # utility per INR
OCCUPANCY = 1.3      # persons per car
FUEL_PER_KM = 4.0    # INR
BASE_PARKING_CBD = 40.0


@dataclass(frozen=True)
class Scenario:
    name: str = "Base"
    bus_priority: bool = False     # dedicated lanes: buses run at free-flow speed
    bus_headway_min: float = 10.0
    cbd_parking_charge: float = 0.0  # extra INR per car trip ending in the CBD


@dataclass
class Result:
    scenario: Scenario
    trips: np.ndarray
    probs: dict[str, np.ndarray]
    link_flow: np.ndarray
    link_time: np.ndarray
    fw_gaps: list[float]
    feedback_rmse: list[float]
    beta: float
    metrics: dict = field(default_factory=dict)


def _intrazonal(net: Network, skim: np.ndarray, link_w: np.ndarray) -> np.ndarray:
    skim = skim.copy()
    for z in range(net.n_nodes):
        skim[z, z] = 0.5 * link_w[net.a == z].min()
    return skim


def _utilities(net, sc: Scenario, car_t, ff_t, dist):
    parking = np.where(net.cbd, BASE_PARKING_CBD + sc.cbd_parking_charge, 0.0)[None, :]
    bus_ivt = 1.25 * (ff_t if sc.bus_priority else car_t)
    bus_ovt = sc.bus_headway_min / 2 + 6.0          # wait + walk access/egress
    return {
        "car": -B_TIME * car_t - B_COST * (FUEL_PER_KM * dist + parking),
        "bus": -0.3 - B_TIME * bus_ivt - 2 * B_TIME * bus_ovt - B_COST * (10 + 1.5 * dist),
        "walk/cycle": -0.4 - 1.6 * B_TIME * (dist / 10.0 * 60),
    }


def run_model(net: Network, sc: Scenario = Scenario(), beta: float | None = None,
              target_trip_km: float = 3.4, n_feedback: int = 8) -> Result:
    P, A = trip_generation(net.population, net.jobs)
    dist = _intrazonal(net, net.shortest_paths(net.length_km)[0], net.length_km)
    ff_t = _intrazonal(net, net.shortest_paths(net.fft_min)[0], net.fft_min)
    car_t = ff_t.copy()

    rmse = []
    for it in range(n_feedback):
        probs, logsum = mode_choice(_utilities(net, sc, car_t, ff_t, dist))
        composite = -logsum / B_TIME             # in car-minute equivalents
        if beta is None:
            beta = calibrate_beta(P, A, composite, target_trip_km, measure=dist)
        T = gravity_model(P, A, composite, beta)
        car_veh = T * probs["car"] / OCCUPANCY
        np.fill_diagonal(car_veh, 0.0)
        flow, gaps = frank_wolfe(net, car_veh)
        t_link = net.link_time(flow)
        new_t = _intrazonal(net, net.shortest_paths(t_link)[0], t_link)
        rmse.append(float(np.sqrt(np.mean((new_t - car_t) ** 2))))
        car_t = car_t + (new_t - car_t) / (it + 1)     # MSA

    res = Result(sc, T, probs, flow, t_link, gaps, rmse, beta)
    res.metrics = _metrics(net, res, dist)
    return res


def _metrics(net, r: Result, dist) -> dict:
    total = r.trips.sum()
    share = {m: round(100 * float((r.trips * p).sum() / total), 1) for m, p in r.probs.items()}
    vc = r.link_flow / net.capacity
    vkt = float(r.link_flow @ net.length_km)
    vht = float(r.link_flow @ r.link_time) / 60
    cbd_dest = r.trips[:, net.cbd].sum()
    return {
        "scenario": r.scenario.name,
        "person_trips": round(float(total)),
        "mode_share_pct": share,
        "car_share_to_cbd_pct": round(100 * float((r.trips * r.probs["car"])[:, net.cbd].sum() / cbd_dest), 1),
        "mean_trip_km": round(float((r.trips * dist).sum() / total), 2),
        "vehicle_km": round(vkt),
        "vehicle_hours": round(vht),
        "avg_network_speed_kmph": round(vkt / vht, 1),
        "links_over_90pct_capacity": int((vc > 0.9).sum()),
        "max_vc": round(float(vc.max()), 2),
        "fw_iterations": len(r.fw_gaps),
        "fw_final_gap": float(f"{r.fw_gaps[-1]:.2e}"),
        "beta": round(r.beta, 4),
    }
