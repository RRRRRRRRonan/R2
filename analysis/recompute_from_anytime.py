#!/usr/bin/env python3
"""Recompute performance tables and Nemenyi rankings from anytime_data.json."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import statistics
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclasses.dataclass
class Record:
    scale: str
    instance_id: str
    method: str
    tardiness: float
    time_sec: float


def load_records(path: pathlib.Path) -> List[Record]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: List[Record] = []
    for item in raw:
        stats_items = item.get("stats") or []
        if not stats_items:
            continue
        last = stats_items[-1]
        records.append(
            Record(
                scale=item["scale"],
                instance_id=item["instance_id"],
                method=item["method"],
                tardiness=float(last["best"]),
                time_sec=float(last["time"]),
            )
        )
    return records


def compute_rankings(per_instance: Dict[str, Dict[str, float]], methods: List[str]) -> dict | None:
    instances = sorted(per_instance.keys())
    if len(instances) < 2 or len(methods) < 2:
        return None
    rank_sums = {m: 0.0 for m in methods}
    for inst in instances:
        values = [per_instance[inst][m] for m in methods]
        ranks = stats.rankdata(values, method="average")
        for m, r in zip(methods, ranks):
            rank_sums[m] += float(r)
    num_instances = len(instances)
    avg_ranks = {m: rank_sums[m] / num_instances for m in methods}
    k = len(methods)
    q_alpha = stats.studentized_range.ppf(1 - 0.05, k, 10_000)
    if math.isnan(q_alpha):
        return None
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * num_instances))
    return {"avg_ranks": avg_ranks, "num_instances": num_instances, "critical_difference": cd}


def write_performance(records: List[Record], output_dir: pathlib.Path) -> None:
    grouped: Dict[str, Dict[str, List[Record]]] = defaultdict(lambda: defaultdict(list))
    best_ref: Dict[tuple[str, str], float] = {}
    for r in records:
        grouped[r.scale][r.method].append(r)
        key = (r.scale, r.instance_id)
        best_ref[key] = min(best_ref.get(key, math.inf), r.tardiness)
    best_positive: Dict[tuple[str, str], float] = {}
    for r in records:
        key = (r.scale, r.instance_id)
        if r.tardiness > 1e-9:
            best_positive[key] = min(best_positive.get(key, math.inf), r.tardiness)

    lines = ["# 性能对比总表（由 anytime_data 重新计算）", ""]
    summary: Dict[str, Dict[str, dict]] = {}
    for scale in sorted(grouped.keys()):
        lines.append(f"## {scale.title()} 实例")
        lines.append("| 策略 | 平均延迟 ± std | 平均时间 (s) | 平均最优性缺口 (%) |")
        lines.append("| --- | --- | --- | --- |")
        summary.setdefault(scale, {})
        for method, recs in sorted(grouped[scale].items()):
            tardiness = [r.tardiness for r in recs]
            runtimes = [r.time_sec for r in recs]
            mean_t = statistics.mean(tardiness)
            std_t = statistics.pstdev(tardiness) if len(tardiness) > 1 else 0.0
            mean_time = statistics.mean(runtimes)
            gaps = []
            for r in recs:
                denom = best_ref[(r.scale, r.instance_id)]
                if denom <= 1e-9:
                    denom = best_positive.get((r.scale, r.instance_id), 1.0)
                gaps.append((r.tardiness - denom) / denom * 100.0)
            mean_gap = statistics.mean(gaps) if gaps else 0.0
            summary[scale][method] = {
                "mean_tardiness": mean_t,
                "std_tardiness": std_t,
                "mean_time": mean_time,
                "mean_gap": mean_gap,
            }
            lines.append(f"| {method} | {mean_t:.2f} ± {std_t:.2f} | {mean_time:.2f} | {mean_gap:.2f} |")
        lines.append("")

    (output_dir / "performance_tables.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "performance_records.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_significance(records: List[Record], output_dir: pathlib.Path) -> None:
    lines = ["# 统计显著性检验（Nemenyi，基于 anytime_data 的最终值）", ""]
    report: Dict[str, dict] = {}
    plots: Dict[str, pathlib.Path] = {}
    for scale in sorted({r.scale for r in records}):
        scale_records = [r for r in records if r.scale == scale]
        per_instance: Dict[str, Dict[str, float]] = defaultdict(dict)
        for r in scale_records:
            per_instance[r.instance_id][r.method] = r.tardiness
        methods = sorted({r.method for r in scale_records})
        complete_methods = [m for m in methods if all(m in per_instance[inst] for inst in per_instance)]
        if len(complete_methods) < 2:
            continue
        ranking = compute_rankings(per_instance, complete_methods)
        if not ranking:
            continue
        lines.append(f"## {scale.title()} 实例")
        lines.append(f"- N={ranking['num_instances']}，临界差异 CD={ranking['critical_difference']:.3f}")
        lines.append("| 算法 | 平均排名 |")
        lines.append("| --- | --- |")
        for method, rank_value in sorted(ranking["avg_ranks"].items(), key=lambda item: item[1]):
            lines.append(f"| {method} | {rank_value:.3f} |")
        lines.append("")
        report[scale] = ranking
        path = output_dir / f"nemenyi_cd_{scale}.png"
        plot_cd_diagram(ranking_info=ranking, title=f"{scale.title()} average ranks (Nemenyi)", path=path)
        plots[scale] = path

    (output_dir / "significance_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "significance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if plots:
        print("Generated CD plots:", ", ".join(f"{k}: {v.name}" for k, v in plots.items()))


def process_dir(input_path: pathlib.Path) -> None:
    records = load_records(input_path)
    write_performance(records, input_path.parent)
    write_significance(records, input_path.parent)
    print(f"updated reports for {input_path.parent}")


def plot_cd_diagram(ranking_info: dict, title: str, path: pathlib.Path) -> None:
    avg_ranks: Dict[str, float] = ranking_info.get("avg_ranks", {})
    if not avg_ranks:
        return
    cd = ranking_info.get("critical_difference")
    if not cd:
        return
    sorted_methods = sorted(avg_ranks.items(), key=lambda item: item[1])
    ranks = [rank for _, rank in sorted_methods]
    labels = [method for method, _ in sorted_methods]
    fig_height = 1.5 + 0.6 * len(labels)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    min_rank = min(ranks) - 0.3
    max_rank = max(ranks) + 0.3
    ax.set_xlim(min_rank, max_rank)
    ax.set_xticks(np.arange(1, len(sorted_methods) + 1))
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    raw_segments = []
    for i in range(len(sorted_methods)):
        start = i
        end = i
        for j in range(i + 1, len(sorted_methods)):
            if ranks[j] - ranks[start] <= cd + 1e-9:
                end = j
            else:
                break
        if end > start:
            raw_segments.append((start, end))
    connectors = []
    for seg in raw_segments:
        if not any(
            other is not seg
            and other[0] <= seg[0]
            and other[1] >= seg[1]
            for other in raw_segments
        ):
            connectors.append(seg)
    connector_levels = max(1, len(connectors))
    connectors_base = len(labels) + 0.3
    cd_y = connectors_base + 0.3 * connector_levels
    top_limit = cd_y + 0.5
    ax.set_ylim(-0.5, top_limit)
    ax.hlines(len(labels) + 0.1, min_rank, max_rank, color="black", linewidth=1)
    for idx, (method, rank_value) in enumerate(sorted_methods):
        y = len(labels) - idx - 0.5
        ax.plot(rank_value, y, "o", color="black")
        ax.text(
            rank_value + 0.1,
            y,
            f"{method} ({rank_value:.2f})",
            va="center",
            fontsize=10,
        )
    for level, (start_idx, end_idx) in enumerate(connectors):
        y = connectors_base + level * 0.3
        ax.plot(
            [ranks[start_idx], ranks[end_idx]],
            [y, y],
            color="gray",
            linewidth=3,
            alpha=0.8,
        )
    ax.plot(
        [min_rank, min_rank + cd],
        [cd_y, cd_y],
        color="black",
        linewidth=2,
    )
    ax.text(min_rank + cd / 2, cd_y + 0.15, f"CD = {cd:.3f}", ha="center")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    base = pathlib.Path("results/analysis")
    candidates = [
        pathlib.Path("results/analysis_small/fixed/anytime_data.json"),
        pathlib.Path("results/analysis_medium/fixed/anytime_data.json"),
        pathlib.Path("results/analysis_large/fixed/anytime_data.json"),
    ]
    for path in candidates:
        if path.exists():
            process_dir(path)


if __name__ == "__main__":
    main()
