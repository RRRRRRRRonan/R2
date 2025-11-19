#!/usr/bin/env python3
"""Aggregate experiment summaries and run significance tests."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
from typing import Dict, List

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistics for experiment groups")
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar=("LABEL", "PATH"),
        help="Label followed by one or more summary directories or files",
    )
    parser.add_argument(
        "--metric",
        default="best_cost",
        help="Metric key in summary.json (supports dot-separated paths)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/significance_report.json"),
        help="Where to store the aggregated statistics",
    )
    return parser.parse_args()


def _iter_summary_files(path: pathlib.Path) -> List[pathlib.Path]:
    if path.is_file() and path.name == "summary.json":
        return [path]
    if path.is_file():
        return []
    return list(path.rglob("summary.json"))


def _load_metric(summary_path: pathlib.Path, metric: str) -> float:
    with summary_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    value = data
    for key in metric.split('.'):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Metric '{metric}' not found in {summary_path}")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Metric '{metric}' in {summary_path} is not numeric")


def collect_group_values(entries: List[str], metric: str) -> List[float]:
    values: List[float] = []
    for entry in entries:
        path = pathlib.Path(entry)
        summaries = _iter_summary_files(path)
        if not summaries:
            raise FileNotFoundError(f"No summary.json files found under {path}")
        for summary in summaries:
            values.append(_load_metric(summary, metric))
    return values


def describe(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = float(np.mean(values)) if values else math.nan
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return {"count": n, "mean": mean, "std": std}


def run_tests(groups: Dict[str, List[float]]):
    results = []
    for (label_a, values_a), (label_b, values_b) in itertools.combinations(groups.items(), 2):
        if len(values_a) < 2 or len(values_b) < 2:
            results.append({
                "pair": f"{label_a} vs {label_b}",
                "t_stat": math.nan,
                "p_value": math.nan,
                "note": "Not enough samples for t-test",
            })
            continue
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        results.append({
            "pair": f"{label_a} vs {label_b}",
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "note": "",
        })
    return results


def main() -> None:
    args = parse_args()
    if not args.group:
        raise SystemExit("Please specify at least one --group")
    groups: Dict[str, List[float]] = {}
    for entry in args.group:
        label = entry[0]
        paths = entry[1:]
        if not paths:
            raise SystemExit(f"Group '{label}' must include at least one path")
        groups[label] = collect_group_values(paths, args.metric)
    group_descriptions = {label: describe(values) for label, values in groups.items()}
    comparisons = run_tests(groups)
    report = {
        "metric": args.metric,
        "groups": group_descriptions,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
