import numpy as np
import pytest

from fourstep import (Network, Scenario, all_or_nothing, calibrate_beta, frank_wolfe,
                      gravity_model, grid_city, mode_choice, run_model, trip_generation)
from fourstep.steps import two_link_equilibrium


def two_link_net():
    """Origin 0 -> destination 1 via two parallel routes (through nodes 2 and 3)."""
    xy = np.array([[0, 0], [2, 0], [1, 1], [1, -1]], float)
    a = np.array([0, 2, 0, 3])
    b = np.array([2, 1, 3, 1])
    fft = np.array([5.0, 0.0001, 8.0, 0.0001])
    cap = np.array([1000.0, 1e9, 2000.0, 1e9])
    z = np.zeros(4)
    return Network(xy, a, b, np.ones(4), fft, cap, np.array(["local"] * 4), z, z, z.astype(bool))


def test_trip_generation_balances():
    P, A = trip_generation(np.array([100.0, 200]), np.array([50.0, 10]))
    assert P.sum() == pytest.approx(A.sum())


def test_gravity_matches_marginals():
    rng = np.random.default_rng(0)
    P, A = rng.uniform(10, 100, 6), rng.uniform(10, 100, 6)
    A *= P.sum() / A.sum()
    T = gravity_model(P, A, rng.uniform(1, 30, (6, 6)), beta=0.1)
    np.testing.assert_allclose(T.sum(1), P, rtol=1e-6)
    np.testing.assert_allclose(T.sum(0), A, rtol=1e-6)


def test_beta_calibration_hits_target():
    rng = np.random.default_rng(1)
    P = A = rng.uniform(10, 100, 8)
    cost = rng.uniform(2, 40, (8, 8))
    beta = calibrate_beta(P, A, cost, target=15.0)
    T = gravity_model(P, A, cost, beta)
    assert (T * cost).sum() / T.sum() == pytest.approx(15.0, abs=1e-3)


def test_logit_probabilities_and_logsum():
    V = {"car": np.array([0.0, -1.0]), "bus": np.array([0.0, -3.0])}
    p, ls = mode_choice(V)
    np.testing.assert_allclose(p["car"] + p["bus"], 1.0)
    assert p["car"][0] == pytest.approx(0.5)
    assert ls[0] == pytest.approx(np.log(2))
    assert ls[1] >= -1.0                       # logsum >= best alternative


def test_all_or_nothing_conserves_flow():
    net = two_link_net()
    D = np.zeros((4, 4))
    D[0, 1] = 500
    f = all_or_nothing(net, D, net.fft_min)
    assert f[0] == 500 and f[2] == 0           # all on the faster route


@pytest.mark.parametrize("demand", [300.0, 1500.0, 4000.0])
def test_frank_wolfe_matches_analytic_equilibrium(demand):
    net = two_link_net()
    D = np.zeros((4, 4))
    D[0, 1] = demand
    flow, gaps = frank_wolfe(net, D, max_iter=2000, gap_tol=1e-7)
    expected = two_link_equilibrium(net.fft_min[[0, 2]], net.capacity[[0, 2]], demand)
    assert flow[0] == pytest.approx(expected, rel=1e-3, abs=0.5)
    assert flow[0] + flow[2] == pytest.approx(demand)


def test_full_model_runs_and_policy_reduces_car_share():
    net = grid_city(n=4)
    base = run_model(net, n_feedback=3)
    assert base.metrics["mean_trip_km"] == pytest.approx(3.4, abs=0.01)
    pricing = run_model(net, Scenario("p", cbd_parking_charge=100), beta=base.beta, n_feedback=3)
    assert pricing.metrics["mode_share_pct"]["car"] < base.metrics["mode_share_pct"]["car"]
    assert base.fw_gaps[-1] < 1e-3
