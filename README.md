# HEA-DRL 实验框架实现

该仓库根据 `docs/` 中的设计文档，实现了一个可运行的 HEA-DRL (Hybrid Evolutionary Algorithm + Deep Reinforcement Learning) 框架。仓库包含：

- `src/`：混合进化算法、移除策略、DRL 代理、数据生成和修复算子的实现。
- `configs/`：对应文档 E1–E5 的实验配置，直接驱动求解和训练流程。
- `main.py`：加载配置并运行 HEA 算法，输出日志、收敛曲线、方案和摘要。
- `train_drl.py`：按照文档 E5 的思路训练轻量级 DRL 移除策略。
- `models/`：示例 DRL 模型文件 (`drl_model.json`)，可直接用于 E3/E4。
- `results/`：运行实验与训练时生成的日志、曲线、方案等输出目录。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行求解实验（以 E2 Shaw 策略为例）：
   ```bash
   python main.py --config configs/E2.yaml
   ```
3. 训练 DRL 策略：
   ```bash
   python train_drl.py --config configs/E5_train.yaml
   ```

运行完成后，可在 `results/<实验名>/` 查看 `run.log`、`convergence.png`、`summary.json` 和 `best_solution.json` 等文件，结合 `docs/` 中的章节进行分析。训练日志和曲线会写入 `results/E5_train/`，并在 `models/` 下生成/覆盖 DRL 模型文件。
