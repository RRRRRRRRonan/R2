#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import re
from typing import List, Dict

import matplotlib.pyplot as plt


def _ensure_matplotlib_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except OSError:
        return


_ensure_matplotlib_dir()


def parse_md(path: pathlib.Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 8 or parts[0] in {"规模", "---"}:
            continue
        scale, method, total, sel, remove, repair, eval_, drl = parts[:8]
        def parse_value(s: str) -> float:
            m = re.match(r"([0-9.]+)", s)
            return float(m.group(1)) if m else 0.0
        rows.append(
            {
                "scale": scale,
                "method": method,
                "T_total": parse_value(total),
                "T_select_cross": parse_value(sel),
                "T_remove": parse_value(remove),
                "T_repair": parse_value(repair),
                "T_eval": parse_value(eval_),
                "T_drl_infer": parse_value(drl),
            }
        )
    return rows


def plot_components(rows: List[Dict[str, float]], out_path: pathlib.Path) -> None:
    if not rows:
        return
    components = [
        ("T_select_cross", "Select+Cross", "#4c72b0"),
        ("T_remove", "Remove", "#dd8452"),
        ("T_repair", "Repair", "#55a868"),
        ("T_eval", "Eval", "#c44e52"),
        ("T_drl_infer", "DRL Inference", "#8172b2"),
    ]
    methods = [r["method"] for r in rows]
    fig, axes = plt.subplots(len(components), 1, figsize=(10, 2.2 * len(components)), sharex=True)
    if len(components) == 1:
        axes = [axes]
    for ax, (comp, label, color) in zip(axes, components):
        values = [r[comp] for r in rows]
        positions = range(len(methods))
        bars = ax.barh(list(positions), values, color=color)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_ylabel(label)
        ax.set_title(f"{label} (s)", loc="left", fontsize=11, pad=6)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01 if values else 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s",
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Runtime (s)")
    axes[0].set_title(f"Runtime breakdown by component ({rows[0]['scale']})")
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot runtime breakdown as per-component bars.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=None,
        help="Path to runtime_breakdown.md (default: auto scan fixed dirs)",
    )
    args = parser.parse_args()
    targets = []
    if args.input:
        targets.append(args.input)
    else:
        for scale in ("small", "medium", "large"):
            path = pathlib.Path(f"results/analysis_{scale}/fixed/runtime_breakdown.md")
            if path.exists():
                targets.append(path)
    for md_path in targets:
        rows = parse_md(md_path)
        if not rows:
            continue
        out_path = md_path.parent / "runtime_breakdown.png"
        plot_components(rows, out_path)
        print(f"plotted {out_path}")


if __name__ == "__main__":
    main()
