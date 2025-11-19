#!/usr/bin/env python3
"""Run the five requested experiments end-to-end and materialize artefacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Sequence

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.config import AlgorithmConfig, ProblemConfig
from src.data import ProblemInstance, generate_random_instance
from src.drl_agent import DRLAgent
from src.hea import HEA
from src.solution import Solution

OUTPUT_DIR = pathlib.Path("results/analysis")
DRL_MODEL = pathlib.Path("models/drl_model.json")


@dataclasses.dataclass
class ScaleSetting:
    name: str
    num_tasks: int
    distribution: str
    population_size: int
    generations: int
    remove_count: int
    seeds: List[int]


@dataclasses.dataclass
class MethodSpec:
    name: str
    kind: str  # "hea", "constructive", "aco", "exact"
    applies_to: Sequence[str] = dataclasses.field(
        default_factory=lambda: ("small", "medium", "large")
    )
    removal_strategy: str | None = None
    remove_override: int | None = None
    seed_offset: int = 0


@dataclasses.dataclass
class ExperimentRecord:
    scale: str
    instance_id: str
    method: str
    tardiness: float
    time_sec: float
    stats: List[tuple[float, float, float]] | None
    timing: dict[str, float]


class ExperimentSuite:
    def __init__(
        self, output_dir: pathlib.Path = OUTPUT_DIR, enable_plots: bool = False
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_plots = enable_plots
        self.scales: Dict[str, ScaleSetting] = {
            "small": ScaleSetting(
                name="small",
                num_tasks=10,
                distribution="uniform",
                population_size=16,
                generations=20,
                remove_count=3,
                seeds=[11, 17],
            ),
            "medium": ScaleSetting(
                name="medium",
                num_tasks=30,
                distribution="uniform",
                population_size=22,
                generations=30,
                remove_count=4,
                seeds=[101, 111],
            ),
            "large": ScaleSetting(
                name="large",
                num_tasks=50,
                distribution="clustered",
                population_size=26,
                generations=30,
                remove_count=5,
                seeds=[201, 211],
            ),
        }
        self.methods: List[MethodSpec] = [
            MethodSpec(name="Exact MIP", kind="exact", applies_to=("small",)),
            MethodSpec(name="Constructive Heuristic", kind="constructive"),
            MethodSpec(
                name="Genetic Algorithm",
                kind="hea",
                removal_strategy="random",
                remove_override=0,
                seed_offset=5,
            ),
            MethodSpec(
                name="Memetic Algorithm",
                kind="hea",
                removal_strategy="shaw",
                seed_offset=7,
            ),
            MethodSpec(name="Ant Colony", kind="aco", seed_offset=9),
            MethodSpec(
                name="HEA-DRL",
                kind="hea",
                removal_strategy="drl",
                seed_offset=11,
            ),
        ]
        self.records: List[ExperimentRecord] = []
        self.transfer_records: List[dict] = []
        self.drl_agent = self._load_drl_agent()

    def _load_drl_agent(self) -> DRLAgent:
        if not DRL_MODEL.exists():
            raise FileNotFoundError(
                "DRL model file models/drl_model.json is required for HEA-DRL experiments"
            )
        return DRLAgent.load(DRL_MODEL)

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[1/6] Running core benchmark experiments...")
        self.records = self._run_core_experiments()
        print("[2/6] Running transfer experiments...")
        self.transfer_records = self._run_transfer_experiments()
        print("[3/6] Generating performance tables...")
        self.generate_performance_tables()
        print("[4/6] Evaluating statistical significance...")
        self.generate_significance_report()
        print("[5/6] Creating anytime and transfer visualisations...")
        self.generate_anytime_plots()
        self.generate_transfer_summary()
        print("[6/6] Detailing runtime breakdown...")
        self.generate_runtime_breakdown()
        print(f"分析完成，输出位于: {self.output_dir}")

    def _run_core_experiments(self) -> List[ExperimentRecord]:
        records: List[ExperimentRecord] = []
        for scale_name, scale in self.scales.items():
            for idx, seed in enumerate(scale.seeds):
                instance = self._build_instance(scale, seed)
                instance_id = f"{scale_name}_I{idx + 1}"
                for method in self.methods:
                    if scale_name not in method.applies_to:
                        continue
                    record = self._dispatch_method(
                        method, scale_name, instance_id, instance, scale, seed
                    )
                    if record:
                        records.append(record)
        return records

    def _build_instance(self, scale: ScaleSetting, seed: int) -> ProblemInstance:
        cfg = ProblemConfig(
            instance_type="random",
            num_tasks=scale.num_tasks,
            distribution=scale.distribution,
            seed=seed,
        )
        return generate_random_instance(cfg)

    def _dispatch_method(
        self,
        method: MethodSpec,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        scale: ScaleSetting,
        seed: int,
    ) -> ExperimentRecord | None:
        if method.kind == "hea":
            return self._run_hea(method, scale_name, instance_id, instance, scale, seed)
        if method.kind == "constructive":
            return self._run_constructive(method.name, scale_name, instance_id, instance)
        if method.kind == "exact":
            return self._run_exact(method.name, scale_name, instance_id, instance)
        if method.kind == "aco":
            return self._run_aco(method.name, scale_name, instance_id, instance, seed)
        raise ValueError(f"Unsupported method kind: {method.kind}")

    # ------------------------------------------------------------------
    # Individual solvers
    # ------------------------------------------------------------------
    def _run_hea(
        self,
        method: MethodSpec,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        scale: ScaleSetting,
        seed: int,
    ) -> ExperimentRecord:
        algo_seed = seed + method.seed_offset
        remove_count = (
            method.remove_override if method.remove_override is not None else scale.remove_count
        )
        algo_cfg = AlgorithmConfig(
            removal_strategy=method.removal_strategy or "random",
            remove_count=remove_count,
            population_size=scale.population_size,
            generations=scale.generations,
            crossover_rate=0.85,
            mutation_rate=0.2,
            elite_rate=0.1,
            model_path=str(DRL_MODEL) if method.removal_strategy == "drl" else None,
            seed=algo_seed,
        )
        logger = logging.getLogger(
            f"suite.{scale_name}.{instance_id}.{method.name.replace(' ', '_')}"
        )
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        agent = self.drl_agent if method.removal_strategy == "drl" else None
        algo = HEA(instance, algo_cfg, logger, agent)
        result = algo.run()
        stats = [(s.elapsed_sec, s.best_cost, s.average_cost) for s in result.stats]
        time_sec = result.timing.get("T_total", 0.0)
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method.name,
            tardiness=result.best_solution.cost(),
            time_sec=time_sec,
            stats=stats,
            timing=result.timing,
        )

    def _run_constructive(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
    ) -> ExperimentRecord:
        start = time.perf_counter()
        order = self._nearest_neighbor_order(instance)
        solution = Solution(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_exact(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
    ) -> ExperimentRecord:
        if len(instance.tasks) > 14:
            raise ValueError("Exact solver is only configured for <=14 tasks")
        start = time.perf_counter()
        order = self._held_karp(instance)
        solution = Solution(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_aco(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        seed: int,
    ) -> ExperimentRecord:
        start = time.perf_counter()
        order = self._ant_colony_opt(instance, seed)
        solution = Solution(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    # ------------------------------------------------------------------
    # Helper heuristics
    # ------------------------------------------------------------------
    def _nearest_neighbor_order(self, instance: ProblemInstance) -> List[int]:
        remaining = set(range(len(instance.tasks)))
        order: List[int] = []
        current: int | None = None
        while remaining:
            def distance_to(idx: int) -> float:
                if current is None:
                    return instance.distance(None, instance.tasks[idx])
                return instance.distance(instance.tasks[current], instance.tasks[idx])

            next_idx = min(remaining, key=distance_to)
            order.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx
        return order

    def _held_karp(self, instance: ProblemInstance) -> List[int]:
        n = len(instance.tasks)
        dist = [[instance.distance(a, b) for b in instance.tasks] for a in instance.tasks]
        depot_dist = [instance.distance(None, task) for task in instance.tasks]
        dp: dict[tuple[int, int], tuple[float, int]] = {}
        for j in range(n):
            mask = 1 << j
            dp[(mask, j)] = (depot_dist[j], -1)
        for mask in range(1, 1 << n):
            for j in range(n):
                if not mask & (1 << j):
                    continue
                if mask == (1 << j):
                    continue
                prev_mask = mask ^ (1 << j)
                best_cost = math.inf
                best_prev = -1
                for k in range(n):
                    if not prev_mask & (1 << k):
                        continue
                    prev_cost = dp[(prev_mask, k)][0] + dist[k][j]
                    if prev_cost < best_cost:
                        best_cost = prev_cost
                        best_prev = k
                dp[(mask, j)] = (best_cost, best_prev)
        full_mask = (1 << n) - 1
        best_cost = math.inf
        best_last = -1
        for j in range(n):
            cost = dp[(full_mask, j)][0] + instance.distance(instance.tasks[j], None)
            if cost < best_cost:
                best_cost = cost
                best_last = j
        order: List[int] = []
        mask = full_mask
        curr = best_last
        while curr != -1:
            order.append(curr)
            _, prev = dp[(mask, curr)]
            mask ^= 1 << curr
            curr = prev
        order.reverse()
        return order

    def _ant_colony_opt(self, instance: ProblemInstance, seed: int) -> List[int]:
        rng = random.Random(seed)
        n = len(instance.tasks)
        pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
        depot_pheromone = [1.0 for _ in range(n)]
        alpha = 1.1
        beta = 2.0
        rho = 0.25
        iterations = 25
        ants = 14
        best_order: List[int] | None = None
        best_cost = math.inf
        for _ in range(iterations):
            iteration_best_order: List[int] | None = None
            iteration_best_cost = math.inf
            for _ in range(ants):
                order = self._construct_ant_route(instance, pheromone, depot_pheromone, rng, alpha, beta)
                cost = Solution(instance, order).cost()
                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_order = order
                if cost < best_cost:
                    best_cost = cost
                    best_order = order
            if iteration_best_order is None:
                continue
            self._evaporate_pheromone(pheromone, depot_pheromone, rho)
            self._deposit_pheromone(
                instance,
                pheromone,
                depot_pheromone,
                iteration_best_order,
                1.0 / max(iteration_best_cost, 1e-6),
            )
        if best_order is None:
            best_order = list(range(n))
        return best_order

    def _construct_ant_route(
        self,
        instance: ProblemInstance,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        rng: random.Random,
        alpha: float,
        beta: float,
    ) -> List[int]:
        remaining = set(range(len(instance.tasks)))
        order: List[int] = []
        current: int | None = None
        while remaining:
            weights: List[tuple[int, float]] = []
            for candidate in remaining:
                if current is None:
                    tau = depot_pheromone[candidate]
                    dist = instance.distance(None, instance.tasks[candidate])
                else:
                    tau = pheromone[current][candidate]
                    dist = instance.distance(instance.tasks[current], instance.tasks[candidate])
                eta = 1.0 / (dist + 1e-6)
                weight = (tau ** alpha) * (eta ** beta)
                weights.append((candidate, weight))
            total = sum(w for _, w in weights)
            if total <= 0:
                chosen = rng.choice(list(remaining))
            else:
                threshold = rng.random() * total
                acc = 0.0
                chosen = weights[-1][0]
                for candidate, weight in weights:
                    acc += weight
                    if acc >= threshold:
                        chosen = candidate
                        break
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return order

    def _evaporate_pheromone(
        self,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        rho: float,
    ) -> None:
        for i in range(len(pheromone)):
            for j in range(len(pheromone)):
                pheromone[i][j] *= (1.0 - rho)
                pheromone[i][j] = max(1e-4, pheromone[i][j])
        for idx in range(len(depot_pheromone)):
            depot_pheromone[idx] *= (1.0 - rho)
            depot_pheromone[idx] = max(1e-4, depot_pheromone[idx])

    def _deposit_pheromone(
        self,
        instance: ProblemInstance,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        order: Sequence[int],
        amount: float,
    ) -> None:
        if not order:
            return
        first = order[0]
        depot_pheromone[first] += amount
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            pheromone[a][b] += amount
            pheromone[b][a] += amount
        depot_pheromone[order[-1]] += amount

    # ------------------------------------------------------------------
    # Transfer experiments
    # ------------------------------------------------------------------
    def _run_transfer_experiments(self) -> List[dict]:
        cases = [
            ("VRPTW", pathlib.Path("data/vrptw_instance.json")),
            ("EVRPTW", pathlib.Path("data/evrptw_instance.json")),
        ]
        results: List[dict] = []
        for name, path in cases:
            if not path.exists():
                raise FileNotFoundError(f"Missing transfer instance: {path}")
            cfg = ProblemConfig(data_file=str(path))
            instance = generate_random_instance(cfg)
            scale = ScaleSetting(
                name=name,
                num_tasks=len(instance.tasks),
                distribution="file",
                population_size=32,
                generations=40,
                remove_count=max(3, len(instance.tasks) // 10),
                seeds=[999],
            )
            baselines = [
                MethodSpec(
                    name="Random Removal HEA",
                    kind="hea",
                    removal_strategy="random",
                    remove_override=scale.remove_count,
                ),
                MethodSpec(name="HEA-DRL", kind="hea", removal_strategy="drl"),
            ]
            for method in baselines:
                record = self._run_hea(
                    method,
                    scale_name=name,
                    instance_id=f"{name}_instance",
                    instance=instance,
                    scale=scale,
                    seed=1234,
                )
                results.append(
                    {
                        "problem": name,
                        "method": method.name,
                        "tardiness": record.tardiness,
                        "time_sec": record.time_sec,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Artefact generation
    # ------------------------------------------------------------------
    def generate_performance_tables(self) -> None:
        best_ref = self._best_reference()
        grouped: Dict[str, Dict[str, List[ExperimentRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self.records:
            grouped[record.scale][record.method].append(record)
        lines = ["# 性能对比总表", ""]
        summary: Dict[str, Dict[str, dict]] = {}
        for scale in ("small", "medium", "large"):
            if scale not in grouped:
                continue
            lines.append(f"## {scale.title()} 实例")
            lines.append("| 策略 | 平均延迟 ± std | 平均时间 (s) | 平均最优性缺口 (%) |")
            lines.append("| --- | --- | --- | --- |")
            summary.setdefault(scale, {})
            for method, records in grouped[scale].items():
                tardiness = [r.tardiness for r in records]
                mean_t = statistics.mean(tardiness)
                std_t = statistics.pstdev(tardiness) if len(tardiness) > 1 else 0.0
                runtimes = [r.time_sec for r in records]
                mean_time = statistics.mean(runtimes)
                gaps = [
                    (r.tardiness - best_ref[(r.scale, r.instance_id)])
                    / best_ref[(r.scale, r.instance_id)]
                    * 100.0
                    for r in records
                ]
                mean_gap = statistics.mean(gaps)
                summary[scale][method] = {
                    "mean_tardiness": mean_t,
                    "std_tardiness": std_t,
                    "mean_time": mean_time,
                    "mean_gap": mean_gap,
                }
                lines.append(
                    f"| {method} | {mean_t:.1f} ± {std_t:.1f} | {mean_time:.2f} | {mean_gap:.2f} |"
                )
            lines.append("")
        (self.output_dir / "performance_tables.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        (self.output_dir / "performance_records.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def generate_significance_report(self) -> None:
        lines = ["# 统计显著性检验", "", "置信水平：95%，采用 Wilcoxon (配对) 与 Friedman (整体) 检验。", ""]
        report: Dict[str, dict] = {}
        for scale in ("small", "medium", "large"):
            records = [r for r in self.records if r.scale == scale]
            if not records:
                continue
            per_instance: Dict[str, Dict[str, float]] = defaultdict(dict)
            for r in records:
                per_instance[r.instance_id][r.method] = r.tardiness
            methods = sorted({r.method for r in records})
            # Filter methods that exist on every instance
            complete_methods = [
                m
                for m in methods
                if all(m in per_instance[inst] for inst in per_instance)
            ]
            if "HEA-DRL" not in complete_methods or len(complete_methods) < 2:
                continue
            lines.append(f"## {scale.title()} 实例")
            wilcoxon: List[dict] = []
            for method in complete_methods:
                if method == "HEA-DRL":
                    continue
                hea_values = []
                baseline_values = []
                for inst in sorted(per_instance.keys()):
                    hea_values.append(per_instance[inst]["HEA-DRL"])
                    baseline_values.append(per_instance[inst][method])
                if len(hea_values) < 2:
                    continue
                stat, p_value = stats.wilcoxon(
                    hea_values, baseline_values, alternative="less"
                )
                lines.append(f"- HEA-DRL vs {method}: p={p_value:.4f}, statistic={stat:.2f}")
                wilcoxon.append(
                    {
                        "baseline": method,
                        "p_value": float(p_value),
                        "statistic": float(stat),
                    }
                )
            # Friedman test across methods with complete data
            matrix = []
            for method in complete_methods:
                matrix.append([per_instance[inst][method] for inst in sorted(per_instance.keys())])
            friedman_stat, friedman_p = stats.friedmanchisquare(*matrix)
            lines.append(
                f"- Friedman χ²={friedman_stat:.3f}, p={friedman_p:.5f}，p<0.05 表示整体差异显著"
            )
            lines.append("")
            report[scale] = {
                "wilcoxon": wilcoxon,
                "friedman": {"statistic": float(friedman_stat), "p_value": float(friedman_p)},
            }
        (self.output_dir / "significance_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        (self.output_dir / "significance_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def generate_anytime_plots(self) -> None:
        target_instance = "medium_I1"
        target_methods = {
            r.method: r
            for r in self.records
            if r.instance_id == target_instance and r.stats is not None
        }
        description = ["# 搜索过程可视化", ""]
        if not target_methods:
            description.append("- 当前运行未捕获逐代统计数据，因此无法绘制曲线。")
        elif self.enable_plots:
            plt.figure(figsize=(7, 4))
            for method, record in target_methods.items():
                elapsed = [item[0] for item in record.stats]
                best = [item[1] for item in record.stats]
                plt.plot(elapsed, best, label=method)
            plt.xlabel("Runtime (s)")
            plt.ylabel("Best tardiness")
            plt.title("Anytime best-cost curve (Medium I1)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "anytime_best_curve.png")
            plt.close()

            plt.figure(figsize=(7, 4))
            for method, record in target_methods.items():
                avg = [item[2] for item in record.stats]
                plt.plot(range(len(avg)), avg, label=method)
            plt.xlabel("Generation")
            plt.ylabel("Average population cost")
            plt.title("Population average cost (Medium I1)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "anytime_population_fitness.png")
            plt.close()

            description.extend(
                [
                    "- `anytime_best_curve.png`: HEA-DRL、GA 与 MA 在代表性实例中的 anytime 性能。",
                    "- `anytime_population_fitness.png`: 对应的平均种群适应度稳定性。",
                ]
            )
        else:
            description.append(
                "- 若需生成 `anytime_best_curve.png` 与 `anytime_population_fitness.png`，请在本地运行"
                " `python analysis/experiment_suite.py --with-plots`。"
            )
        (self.output_dir / "anytime_summary.md").write_text(
            "\n".join(description) + "\n", encoding="utf-8"
        )

    def generate_transfer_summary(self) -> None:
        lines = ["# 策略迁移实验", ""]
        lines.append("| 问题 | 策略 | 总延迟 | 运行时间 (s) |")
        lines.append("| --- | --- | --- | --- |")
        for record in self.transfer_records:
            lines.append(
                f"| {record['problem']} | {record['method']} | {record['tardiness']:.1f} | {record['time_sec']:.2f} |"
            )
        lines.append("")
        if self.enable_plots:
            lines.append(
                "> 若本地启用了 `--with-plots`，会同时生成 `transfer_tardiness.png` 展示延迟与耗时对比。"
            )
        else:
            lines.append(
                "> 当前运行跳过了 `transfer_tardiness.png` 绘制，需在本地执行 `python analysis/experiment_suite.py --with-plots` 才会生成。"
            )
        (self.output_dir / "transfer_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        (self.output_dir / "transfer_summary.json").write_text(
            json.dumps(self.transfer_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not self.transfer_records:
            return
        if not self.enable_plots:
            return

        problems = list(dict.fromkeys(record["problem"] for record in self.transfer_records))
        methods = list(dict.fromkeys(record["method"] for record in self.transfer_records))
        tardiness = {
            (rec["problem"], rec["method"]): rec["tardiness"] for rec in self.transfer_records
        }
        runtimes = {
            (rec["problem"], rec["method"]): rec["time_sec"] for rec in self.transfer_records
        }
        positions = np.arange(len(problems))
        width = 0.8 / max(1, len(methods))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, metric_name, data in zip(
            axes,
            ["Total tardiness", "Runtime (s)"],
            [tardiness, runtimes],
        ):
            for idx, method in enumerate(methods):
                offsets = positions - 0.4 + width / 2 + idx * width
                axis.bar(
                    offsets,
                    [data.get((problem, method), 0.0) for problem in problems],
                    width=width,
                    label=method,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(problems)
            axis.set_title(metric_name)
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.2), ncol=len(methods))
        fig.suptitle("Transfer performance comparison")
        fig.tight_layout()
        plt.savefig(self.output_dir / "transfer_tardiness.png")
        plt.close(fig)

    def generate_runtime_breakdown(self) -> None:
        components = [
            "T_select_cross",
            "T_remove",
            "T_repair",
            "T_eval",
            "T_drl_infer",
        ]
        lines = ["# 运行时间与开销分析", ""]
        lines.append("| 规模 | 策略 | 总时间 (s) | 选择+交叉 | 移除 | 修复 | 评估 | DRL 推理 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        plot_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for scale in ("small", "medium", "large"):
            for method in ["Genetic Algorithm", "Memetic Algorithm", "HEA-DRL"]:
                subset = [
                    r
                    for r in self.records
                    if r.scale == scale and r.method == method and r.timing
                ]
                if not subset:
                    continue
                totals = {
                    comp: statistics.mean([rec.timing.get(comp, 0.0) for rec in subset])
                    for comp in components
                }
                total_time = statistics.mean([rec.timing.get("T_total", 0.0) for rec in subset])
                plot_data[scale][method] = {**totals, "T_total": total_time}
                def fmt(name: str) -> str:
                    value = totals.get(name, 0.0)
                    return f"{value:.2f} ({value / total_time:.1%})" if total_time else "0"

                lines.append(
                    f"| {scale} | {method} | {total_time:.2f} | {fmt('T_select_cross')} | {fmt('T_remove')} | "
                    f"{fmt('T_repair')} | {fmt('T_eval')} | {fmt('T_drl_infer')} |"
                )
        lines.append("")
        if self.enable_plots:
            lines.append(
                "> 启用 `--with-plots` 时会额外输出 `runtime_breakdown.png` 展示堆叠柱状图。"
            )
        else:
            lines.append(
                "> 当前运行未生成 `runtime_breakdown.png`，可在本地使用 `python analysis/experiment_suite.py --with-plots` 来绘制。"
            )
        (self.output_dir / "runtime_breakdown.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        if self.enable_plots and plot_data:
            scales = list(plot_data.keys())
            fig, axes = plt.subplots(1, len(scales), figsize=(5 * len(scales), 4), sharey=True)
            if len(scales) == 1:
                axes = [axes]
            for ax, scale in zip(axes, scales):
                methods = list(plot_data[scale].keys())
                positions = np.arange(len(methods))
                bottoms = np.zeros(len(methods))
                for comp in components:
                    values = np.array([plot_data[scale][method].get(comp, 0.0) for method in methods])
                    ax.bar(positions, values, bottom=bottoms, label=comp)
                    bottoms += values
                ax.set_title(f"{scale.title()} instances")
                ax.set_xticks(positions)
                ax.set_xticklabels(methods, rotation=20, ha="right")
                ax.set_ylabel("Runtime (s)")
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
            plt.tight_layout()
            plt.savefig(self.output_dir / "runtime_breakdown.png")
            plt.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _best_reference(self) -> Dict[tuple[str, str], float]:
        best: Dict[tuple[str, str], float] = {}
        for record in self.records:
            key = (record.scale, record.instance_id)
            best[key] = min(best.get(key, math.inf), record.tardiness)
        return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HEA-DRL experiment suite")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="目录，用于保存分析产物",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="本地运行时启用，以生成 PNG 图像 (CI/在线环境默认跳过)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = ExperimentSuite(output_dir=args.output, enable_plots=args.with_plots)
    suite.run()


if __name__ == "__main__":
    main()
