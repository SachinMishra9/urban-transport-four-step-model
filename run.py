"""Reproduce the README results.

    python run.py           # ~40 s
    python run.py --quick   # CI smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fourstep import Scenario, grid_city, run_model
from fourstep.plot import plot_convergence, plot_mode_shares, plot_vc_maps

SCENARIOS = [
    Scenario("Base"),
    Scenario("Bus priority lanes + 5-min headway", bus_priority=True, bus_headway_min=5.0),
    Scenario("CBD parking charge (+INR 100)", cbd_parking_charge=100.0),
    Scenario("Combined package", bus_priority=True, bus_headway_min=5.0, cbd_parking_charge=100.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    net = grid_city(n=4 if a.quick else 6)
    feedback = 2 if a.quick else 8
    base = run_model(net, SCENARIOS[0], n_feedback=feedback)
    # beta is calibrated once on the base year and then held fixed for policy tests
    results = [base] + [run_model(net, s, beta=base.beta, n_feedback=feedback) for s in SCENARIOS[1:]]

    (out / "metrics.json").write_text(json.dumps([r.metrics for r in results], indent=2))
    plot_vc_maps(net, [results[0], results[-1]], str(out / "congestion_map.png"))
    plot_mode_shares(results, str(out / "mode_shares.png"))
    plot_convergence(results, str(out / "convergence.png"))

    for r in results:
        m = r.metrics
        print(f"{m['scenario']:38s} car {m['mode_share_pct']['car']:5.1f}%  bus {m['mode_share_pct']['bus']:5.1f}%"
              f"  active {m['mode_share_pct']['walk/cycle']:5.1f}%  veh-h {m['vehicle_hours']:6,}"
              f"  veh-km {m['vehicle_km']:7,}  links>90% {m['links_over_90pct_capacity']:2d}"
              f"  maxV/C {m['max_vc']:.2f}  car->CBD {m['car_share_to_cbd_pct']}%")


if __name__ == "__main__":
    main()
