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

## 分析套件：一次生成五类实验佐证

文档中提出的五项报告需求（多基线性能、统计显著性、搜索过程可视化、跨问题迁移、运行时间拆解）已封装在 `analysis/experiment_suite.py` 中。脚本会自动生成小/中/大型实例，调用 Exact DP、构造式启发式、GA、MA、ACO 与 HEA-DRL 等求解流程并汇总结果：

```bash
python analysis/experiment_suite.py
```

脚本会逐个运行上述方法，并将产物写入 `results/analysis/`，包括：

- `performance_tables.md`：小/中/大规模实例与 Exact MIP、GA、MA、ACO、构造式启发式的量化对比；
- `significance_report.(md|json)`：HEA-DRL 对各基线的 Wilcoxon p 值以及全体方法的 Friedman 检验；
- `anytime_summary.md`、`transfer_summary.md`、`runtime_breakdown.md`：针对搜索过程、迁移实验、运行时开销的文字/表格摘要；

> 默认不会在仓库内生成 PNG 图像，避免阻塞远程 PR。若需在本地绘制 `anytime_best_curve.png`、`transfer_tardiness.png`、`runtime_breakdown.png` 等图表，请运行 `python analysis/experiment_suite.py --with-plots`（同样支持 `--output` 来自定义输出目录）。
