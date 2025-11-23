#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib


def _ensure_matplotlib_dir() -> None:
    """Force Matplotlib to use a writable cache dir."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except OSError:
        return


_ensure_matplotlib_dir()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot anytime average curves from saved JSON stats."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("results/analysis_small/anytime_data.json"),
        help="Path to anytime_data.json",
    )
    parser.add_argument(
        "--scale",
        default="small",
        help="Scale name to plot (e.g., small, medium)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to save the PNG (default: <input parent>/replot)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=200,
        help="Number of points for interpolation grid",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional: override max runtime shown on x-axis",
    )
    parser.add_argument(
        "--plateau-midpoint",
        action="store_true",
        help="Optional: set x-axis limit so plateau starts around the midpoint",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.01,
        help="Relative tolerance to detect plateau start (default 1%% of final value)",
    )
    return parser.parse_args()


def replot_anytime(
    input_path: pathlib.Path,
    scale: str,
    output_dir: pathlib.Path,
    num_points: int = 200,
    max_time_override: float | None = None,
    plateau_midpoint: bool = False,
    plateau_threshold: float = 0.01,
) -> pathlib.Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    records = [d for d in data if d.get("scale") == scale and d.get("stats")]
    if not records:
        raise SystemExit(f"No stats for scale '{scale}' found in {input_path}")

    methods = sorted({rec["method"] for rec in records})
    if max_time_override is not None and max_time_override > 0:
        max_time = max_time_override
    else:
        max_time = max(max(point["time"] for point in rec["stats"]) for rec in records)
    if max_time <= 0:
        raise SystemExit("Non-positive max_time, cannot plot")

    grid = np.linspace(0.0, max_time, num_points)
    plt.figure(figsize=(13, 5.5))
    plotted = False
    mean_curves: dict[str, np.ndarray] = {}
    for method in methods:
        curves = []
        for rec in records:
            if rec["method"] != method:
                continue
            t = np.array([item["time"] for item in rec["stats"]], dtype=float)
            y = np.array([item["best"] for item in rec["stats"]], dtype=float)
            if t.size < 2:
                continue
            curves.append(np.interp(grid, t, y, left=y[0], right=y[-1]))
        if curves:
            plt.plot(grid, np.mean(curves, axis=0), label=method)
            mean_curves[method] = np.mean(curves, axis=0)
            plotted = True

    if not plotted:
        raise SystemExit("No curves could be plotted (missing stats?)")

    plt.xlabel("Runtime (s)")
    plt.ylabel("Best tardiness (avg over seeds)")
    plt.title(f"Anytime average curves ({scale.title()})")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if plateau_midpoint and mean_curves:
        plateau_start = 0.0
        for mean_y in mean_curves.values():
            final = mean_y[-1]
            tol = abs(final) * plateau_threshold
            for idx, value in enumerate(mean_y):
                if value <= final + tol:
                    plateau_start = max(plateau_start, grid[idx])
                    break
        if plateau_start > 0:
            plt.xlim(0, plateau_start * 2)
    plt.tight_layout(rect=(0, 0, 0.93, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"anytime_avg_{scale}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    args = _parse_args()
    scale = args.scale.lower()
    output_dir = args.output_dir or (args.input.parent / "replot")
    out_path = replot_anytime(
        args.input,
        scale,
        output_dir,
        num_points=args.points,
        max_time_override=args.max_time,
        plateau_midpoint=args.plateau_midpoint,
        plateau_threshold=args.plateau_threshold,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
