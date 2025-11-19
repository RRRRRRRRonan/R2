from __future__ import annotations

import argparse
import logging
import pathlib
import random
from typing import List

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

from src.config import load_config
from src.data import generate_random_instance
from src.drl_agent import DRLAgent
from src.repair import greedy_repair
from src.solution import random_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DRL removal agent")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args()


def setup_logger(output_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def evaluate_weights(weights: List[float], instance, remove_count: int, samples: int = 5) -> float:
    agent = DRLAgent(list(weights))
    rewards = []
    for _ in range(samples):
        solution = random_solution(instance)
        base_cost = solution.cost()
        selected = agent.select_tasks(instance, solution.order, remove_count)
        removed = solution.remove_tasks(selected)
        greedy_repair(solution, removed)
        rewards.append(base_cost - solution.cost())
    return sum(rewards) / max(len(rewards), 1)


def run_training(config_path: pathlib.Path) -> None:
    config = load_config(config_path)
    instance = generate_random_instance(config.problem)
    output_dir = pathlib.Path(config.logging.output_dir)
    logger = setup_logger(output_dir)
    remove_count = config.algorithm.remove_count
    weights = [0.0] * 5
    best_reward = -float("inf")
    history: List[tuple[int, float]] = []

    for iteration in range(1, config.training.iterations + 1):
        noise = [random.uniform(-config.training.noise_scale, config.training.noise_scale) for _ in weights]
        candidate_weights = [w + config.training.learning_rate * n for w, n in zip(weights, noise)]
        reward = evaluate_weights(candidate_weights, instance, remove_count)
        if reward > best_reward:
            weights = candidate_weights
            best_reward = reward
        if iteration % config.training.validation_interval == 0:
            val = evaluate_weights(weights, instance, remove_count, samples=8)
            history.append((iteration, val))
            logger.info("Iter %s | avg_reward=%.4f", iteration, val)

    agent = DRLAgent(weights)
    model_path = pathlib.Path(config.algorithm.model_path or "models/drl_model.json")
    agent.save(model_path)
    logger.info("Training completed. Model saved to %s", model_path)
    save_training_curve(history, output_dir)


def save_training_curve(history: List[tuple[int, float]], output_dir: pathlib.Path) -> None:
    if not history or plt is None:
        return
    iterations = [item[0] for item in history]
    rewards = [item[1] for item in history]
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, rewards, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Average reward")
    plt.title("DRL training curve")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)
