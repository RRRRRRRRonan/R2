# 策略迁移实验

| 问题 | 策略 | 总延迟 | 运行时间 (s) |
| --- | --- | --- | --- |
| VRPTW | Random Removal HEA | 34365.1 | 11.45 |
| VRPTW | HEA-DRL | 37233.1 | 11.89 |
| EVRPTW | Random Removal HEA | 7315.2 | 1.63 |
| EVRPTW | HEA-DRL | 7667.9 | 1.68 |

> 当前运行跳过了 `transfer_tardiness.png` 绘制，需在本地执行 `python analysis/experiment_suite.py --with-plots` 才会生成。
