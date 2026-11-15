"""Supplementary charts from the raw run files (results/charts/*.png).

Palette: the validated categorical slots 1-4 (blue, orange, aqua, yellow) on the light chart surface.
One axis per chart; two measures of different scale use small multiples, never a dual axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import ROOT_DIR

RUNS = Path(ROOT_DIR) / "results" / "runs"
OUT = Path(ROOT_DIR) / "results" / "charts"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.7)
    ax.set_axisbelow(True)


def queue_chart() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    style(ax)
    for key, label, col in [("burst", "Burst", ORANGE), ("overload", "Sustained overload", BLUE),
                            ("rpm_halved", "Per-minute limit halved", AQUA), ("steady", "Steady", YELLOW)]:
        m = json.loads((RUNS / key / "run_meta.json").read_text(encoding="utf-8"))
        ts = m["queue_timeseries"]
        ax.plot([r[0] for r in ts], [r[1] for r in ts], color=col, linewidth=2, label=label)
    ax.axhline(40, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2, 41, "queue capacity (40)", color=MUTED, fontsize=8)
    ax.set_xlabel("seconds since first request (stand-in)", color=MUTED, fontsize=9)
    ax.set_ylabel("requests waiting", color=MUTED, fontsize=9)
    ax.set_title("Queue length over time", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    p = OUT / "queue_length.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    return p


def retry_chart() -> Path:
    keys = [("retry_none", "No retries", BLUE), ("retry_naive", "Naive retries", ORANGE),
            ("retry_relayflow", "RelayFlow policy", AQUA)]
    data = {}
    for k, _l, _c in keys:
        m = json.loads((RUNS / k / "run_meta.json").read_text(encoding="utf-8"))
        ev = [json.loads(x) for x in (RUNS / k / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        data[k] = {"load": m["provider_stats"]["arrivals"], "rejected": m["provider_stats"]["rejected_rpm"],
                   "ok": 100 * sum(e["status"] == "ok" for e in ev) / len(ev),
                   "p95": sorted(e["first_token_ms"] for e in ev if e["status"] == "ok" and e["first_token_ms"])[
                       int(0.95 * (sum(e["status"] == "ok" and e["first_token_ms"] is not None for e in ev) - 1))] / 1000}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), facecolor=SURFACE)
    for ax, (field, title, unit) in zip(axes, [("load", "Requests reaching provider", ""),
                                               ("ok", "Succeeded", "%"), ("p95", "p95 first token", " s")]):
        style(ax)
        for i, (k, label, col) in enumerate(keys):
            ax.bar(i, data[k][field], color=col, width=0.62)
            ax.text(i, data[k][field], f"{data[k][field]:.0f}{unit}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(3), ["none", "naive", "Relay\nFlow"], fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.set_ylim(0, max(data[k][field] for k, _l, _c in keys) * 1.2)
    fig.suptitle("Same overloaded stand-in provider, three client retry behaviours", x=0.01, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    p = OUT / "retry_comparison.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (queue_chart(), retry_chart()):
        print("wrote", p)
