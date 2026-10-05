# 复现_v0.4（v0.4 PDF 数字复现包）

来源：`../scheduler/` 现行 10 个 py + `../tables.npz` 的原样拷贝（F1~F7 已入库状态），不含大文件。
数据另取：`../data/era5_ras_zhoushan_2024.nc`（3GB，不进包）。

## 跑法（工作目录=本文件夹）

```powershell
python repro_v04.py
```

依赖：numpy / scipy / netCDF4（读 .nc 需 netCDF4）。

## 预期数字（2026-10-06 实测复现）

合成基准：B0 8.3667t / B1 6.0240% / B2 2.9522% / ours 6.9331%，pp +0.9091。
真实风 2024（8784h）：B0 15459.9926t / B1 3.0619% / ours 3.5435%，pp +0.4815；
B2 在现行多机转正代码下为 -1.67%（v3 的 -0.95% 系转正前残留，pp 不受影响）。

另可跑自测试：`python -m scheduler.test_scheduler`（工作目录=本文件夹）。
