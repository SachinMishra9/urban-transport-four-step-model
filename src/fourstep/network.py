"""Road network and zoning for a synthetic grid city.

Every grid node is a traffic analysis zone (TAZ) centroid. Links are
directed; a ring road and two radial arterials are faster and wider than
local streets. Link travel time follows the BPR volume-delay function
    t = t0 * (1 + alpha * (v / c) ** beta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ALPHA, BETA = 0.15, 4.0


@dataclass
class Network:
    xy: np.ndarray          # (N, 2) node coordinates, km
    a: np.ndarray           # (L,) link tail node
    b: np.ndarray           # (L,) link head node
    length_km: np.ndarray   # (L,)
    fft_min: np.ndarray     # (L,) free-flow time
    capacity: np.ndarray    # (L,) veh / hour
    kind: np.ndarray        # (L,) "local" | "arterial" | "ring"
    population: np.ndarray  # (N,)
    jobs: np.ndarray        # (N,)
    cbd: np.ndarray         # (N,) bool

    @property
    def n_nodes(self) -> int:
        return len(self.xy)

    @property
    def n_links(self) -> int:
        return len(self.a)

    def link_time(self, flow: np.ndarray) -> np.ndarray:
        return self.fft_min * (1 + ALPHA * (flow / self.capacity) ** BETA)

    def _graph(self, weight: np.ndarray) -> csr_matrix:
        return csr_matrix((weight, (self.a, self.b)), shape=(self.n_nodes, self.n_nodes))

    def shortest_paths(self, weight: np.ndarray):
        """All-pairs shortest path costs and predecessor matrix."""
        return dijkstra(self._graph(weight), directed=True, return_predecessors=True)

    def link_index(self) -> dict[tuple[int, int], int]:
        return {(int(u), int(v)): k for k, (u, v) in enumerate(zip(self.a, self.b))}


def grid_city(n: int = 6, spacing_km: float = 1.5, seed: int = 0) -> Network:
    rng = np.random.default_rng(seed)
    xs = np.arange(n) * spacing_km
    xy = np.array([(x, y) for y in xs for x in xs])
    centre = xy.mean(0)
    r = np.hypot(*(xy - centre).T)
    mid = (n - 1) / 2

    def node(i, j):
        return j * n + i

    ring = {k for k in range(n * n) if min(k % n, k // n) == 1 or max(k % n, k // n) == n - 2}
    radial_row = {int(np.floor(mid)), int(np.ceil(mid))}

    a, b, kind = [], [], []
    for j in range(n):
        for i in range(n):
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= n or jj >= n:
                    continue
                u, v = node(i, j), node(ii, jj)
                if (dj == 0 and j in radial_row) or (di == 0 and i in radial_row):
                    k = "arterial"
                elif u in ring and v in ring and (
                        (dj == 0 and j in (1, n - 2)) or (di == 0 and i in (1, n - 2))):
                    k = "ring"
                else:
                    k = "local"
                a += [u, v]
                b += [v, u]
                kind += [k, k]
    a, b, kind = np.array(a), np.array(b), np.array(kind)
    length = np.hypot(*(xy[a] - xy[b]).T)
    speed = np.select([kind == "ring", kind == "arterial"], [50.0, 40.0], 25.0)
    cap = np.select([kind == "ring", kind == "arterial"], [2400.0, 1600.0], 600.0)
    fft = length / speed * 60

    # jobs concentrate in the centre, people live further out
    jobs = np.round(20000 * np.exp(-(r ** 2) / (2 * 1.6 ** 2)) + 600 * rng.uniform(0.5, 1.5, len(xy)))
    population = np.round(12000 * (0.5 + r / r.max()) * rng.uniform(0.8, 1.2, len(xy)))
    cbd = r <= 1.6
    return Network(xy, a, b, length, fft, cap, kind, population, jobs, cbd)
