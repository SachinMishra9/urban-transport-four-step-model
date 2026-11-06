from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from .network import Network  # noqa: E402


def plot_vc_maps(net: Network, results, path: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 5.4), squeeze=False)
    norm = Normalize(0, 1.3)
    cmap = plt.get_cmap("RdYlGn_r")
    for ax, r in zip(axes[0], results):
        segs, cols, widths = [], [], []
        for k in range(net.n_links):
            p, q = net.xy[net.a[k]], net.xy[net.b[k]]
            d = q - p
            off = np.array([d[1], -d[0]]) / np.hypot(*d) * 0.09   # drive on the left: offset
            segs.append([p + off, q + off])
            cols.append(cmap(norm(r.link_flow[k] / net.capacity[k])))
            widths.append({"ring": 3.2, "arterial": 2.4}.get(net.kind[k], 1.3))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths))
        size = 6 + net.jobs / net.jobs.max() * 160
        ax.scatter(*net.xy.T, s=size, c=np.where(net.cbd, "#1f4e9c", "#999"), zorder=3, alpha=0.8)
        m = r.metrics
        ax.set_title(f"{m['scenario']}\ncar share {m['mode_share_pct']['car']}% | "
                     f"{m['vehicle_hours']:,} veh-h | {m['links_over_90pct_capacity']} links > 90% cap",
                     fontsize=9.5)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.75, pad=0.02, label="volume / capacity (AM peak)")
    fig.text(0.42, 0.1, "dot size = jobs, blue = CBD zone, thick = ring road / arterials",
             ha="center", fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mode_shares(results, path: str) -> None:
    names = [r.metrics["scenario"] for r in results]
    modes = list(results[0].metrics["mode_share_pct"])
    colours = {"car": "#d1495b", "bus": "#1f4e9c", "walk/cycle": "#2a9d55"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(results))
    for m in modes:
        v = np.array([r.metrics["mode_share_pct"][m] for r in results])
        ax.barh(names, v, left=left, color=colours.get(m), label=m)
        for i, (lft, val) in enumerate(zip(left, v)):
            ax.text(lft + val / 2, i, f"{val:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += v
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of AM-peak person trips (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    ax.set_title("Mode split by policy scenario")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in results:
        a1.semilogy(r.fw_gaps, label=r.metrics["scenario"], lw=1.2)
        a2.semilogy(range(1, len(r.feedback_rmse) + 1), r.feedback_rmse, "o-", ms=3,
                    label=r.metrics["scenario"])
    a1.set_title("Frank-Wolfe assignment (final feedback loop)")
    a1.set_xlabel("iteration")
    a1.set_ylabel("relative gap")
    a2.set_title("Demand-supply feedback (MSA)")
    a2.set_xlabel("feedback iteration")
    a2.set_ylabel("RMSE change in car skim (min)")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
