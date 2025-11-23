# Transfer scenarios (placeholders)

以下为 4 个迁移场景的占位数据与标题，便于后续替换为真实基准文件：

- Solomon/Homberger VRPTW：`data/transfer_scenarios/solomon_c101_sample.json`、`data/transfer_scenarios/homberger_h1_sample.json`
- Li & Lim EVRPTW/EVRPNL：`data/transfer_scenarios/lilim_evrptw_sample.json`
- Schneider E-VRPTW (充电站)：可先用 `lilim_evrptw_sample.json`，替换为 Schneider 实例后同路径
- Uchoa CVRPTW/PDPTW：`data/transfer_scenarios/uchoa_cvrptw_pdptw_sample.json`

上述 JSON 仅为占位小样例（字段与现有 VRPTW 示例一致），正式实验请替换为对应基准的完整数据，并在 `_run_transfer_experiments` 或命令行中指向真实文件。实验标题可直接用场景名：

1) Solomon/Homberger VRPTW
2) Li & Lim EVRPTW/EVRPNL
3) Schneider E-VRPTW
4) Uchoa CVRPTW/PDPTW
