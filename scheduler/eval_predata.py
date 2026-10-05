"""数据前评估：只验证流程与误差方法，不产出任何节油结论。
占位参数下跑通：插值误差 / 网格敏感 / 换档计数 / 内存 / 越界fallback / 耗时。
用法：python -m scheduler.eval_predata（需先有 tables.npz）
口径声明：本文件为单台口径方法演示（真值穷举用 models.feasible 单机1500kW），
与主流程多机口径（build_tables.feasible_multi / dg_opt，4500kW）分叉，数值仅作方法演示，不产结论。
"""
import time
import numpy as np
from . import config as C
from .models import total_fuel_rate, feasible
from .online import OnlineController
from .synth import make_voyage


def interp_error(n_samples=200, seed=0):
    ctl = OnlineController("tables.npz")
    rng = np.random.default_rng(seed)
    ws = rng.uniform(C.WS_GRID[0], C.WS_GRID[-1], n_samples)
    wa = rng.uniform(C.WA_GRID[0], C.WA_GRID[-1], n_samples)
    vs = rng.uniform(C.VS_GRID[0], C.VS_GRID[-1], n_samples)
    errs_n, errs_f = [], []
    for wsv, wav, vsk in zip(ws, wa, vs):
        v_ms = vsk * 1852.0 / 3600.0
        di, ti = ctl.d, ctl.t  # 固定当前档，只看精调插值（不触发换档逻辑）
        d, t = C.D_VALS[di], C.T_VALS[ti]
        n_tab = float(ctl._fn([(wsv, wav, vsk, di, ti)])[0])
        # 同档位下直接穷举最优（真值；单台口径方法演示，用feasible单机1500kW）
        best, bn = 1e18, 0.0
        for n in C.N_GRID:
            f, p_me, p_dg = total_fuel_rate(v_ms, wsv, n, d, t)
            if feasible(p_me, p_dg) and f < best:
                best, bn = f, n
        f_tab, _, _ = total_fuel_rate(v_ms, wsv, n_tab, d, t)
        errs_n.append(abs(n_tab - bn))
        errs_f.append(abs(f_tab - best) / max(best, 1e-12))
    return float(np.mean(errs_n)), float(np.max(errs_n)), \
        float(np.mean(errs_f)), float(np.max(errs_f))


def grid_sensitivity():
    # 转速网格 25 vs 13 点：同格点最优差（占位下的方法演示）
    coarse = np.linspace(0.0, C.N_MAX, 13)
    diffs = []
    for ws in C.WS_GRID[::2]:
        for vs_kn in C.VS_GRID:
            v_ms = vs_kn * 1852.0 / 3600.0
            for d_idx in range(3):
                for t_idx in range(3):
                    d, t = C.D_VALS[d_idx], C.T_VALS[t_idx]
                    bf = [(total_fuel_rate(v_ms, ws, n, d, t)[0], n)
                          for n in C.N_GRID
                          if feasible(*total_fuel_rate(v_ms, ws, n, d, t)[1:])]
                    cf = [(total_fuel_rate(v_ms, ws, n, d, t)[0], n)
                          for n in coarse
                          if feasible(*total_fuel_rate(v_ms, ws, n, d, t)[1:])]
                    if bf and cf:
                        diffs.append(abs(min(bf)[1] - min(cf)[1]))
    return float(np.mean(diffs)), float(np.max(diffs))


def gear_switches():
    voyage, _ = make_voyage()
    ctl = OnlineController("tables.npz")
    seen, switches, prev = set(), 0, None
    for wsv, wav, vsk in voyage:
        _, _, d, t, _ = ctl.step(wsv, wav, vsk)
        seen.add((d, t))
        if prev is not None and (d, t) != prev:
            switches += 1
        prev = (d, t)
    return len(seen), switches, len(voyage)


def memory_estimate():
    import os
    z = np.load("tables.npz")
    n, p = z["n_star"], z["pme_star"]
    raw = n.nbytes + p.nbytes
    disk = os.path.getsize("tables.npz")
    return n.shape, raw, disk


def fallback_probe():
    ctl = OnlineController("tables.npz")
    outs = []
    for wsv, wav, vsk in [(-5, -10, 5), (0, 0, 10), (30, 200, 20)]:
        n, p, d, t, _ = ctl.step(wsv, wav, vsk)
        outs.append((round(float(n), 1), round(float(p), 1), d, t))
    z = np.load("tables.npz")
    frac0 = float((z["n_star"] == 0).mean())
    return outs, frac0


def timing(n=2000):
    ctl = OnlineController("tables.npz")
    t0 = time.perf_counter()
    for i in range(n):
        ctl.step(11.0, 60.0, 12.0)
    dt = (time.perf_counter() - t0) / n * 1000.0
    return dt


def main():
    en_mean, en_max, ef_mean, ef_max = interp_error()
    g_mean, g_max = grid_sensitivity()
    gears, switches, steps = gear_switches()
    shape, raw, disk = memory_estimate()
    outs, frac0 = fallback_probe()
    ms = timing()
    print("== predata eval (placeholder params, method only) ==")
    print(f"interp |dn|: mean={en_mean:.2f}rpm max={en_max:.2f}rpm |df|/f: mean={ef_mean:.4f} max={ef_max:.4f}")
    print(f"grid N25vsN13 |dn|: mean={g_mean:.2f}rpm max={g_max:.2f}rpm")
    print(f"gears: visited={gears} switches={switches}/{steps}")
    print(f"table: shape={shape} raw={raw/1024:.1f}KB disk={disk/1024:.1f}KB")
    print(f"fallback(oob->clamp): {outs} | n*=0 frac={frac0:.2f}")
    print(f"lookup: {ms:.3f} ms/step (PC+scipy, not STM32)")


if __name__ == "__main__":
    main()
