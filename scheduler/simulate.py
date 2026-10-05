"""时间序列回放仿真：B0无帆 / B2常规 / 本方案，同输入跑分。
指标：全程总节油率（相对B0）、DG低效区时间占比、在线单步耗时。

多机转正版（2026-10-02，dg转正实验）：
- 每步调dg.dg_opt在线执行开停，启动/切换代价计入DG油耗（fuel_extra_g）。
  运行油耗用dg.fuel_dg_rate(p_dg, n_new)，ME部分不变。
- 四策略各维护独立n_online（初值1），切换数/过载数/附加油耗随rep返回。
- 表结构不动：查表仍只给帆转速n*，开机数由P_DG经dg_opt派生。
- 新增参数无（全复用dg.py占位参数，TODO见dg.py）。
"""
import numpy as np
from . import config as C
from .models import total_fuel_rate, eta_patent, sfoc_me, sfoc_dg, hull_resistance
from .build_tables import segment
from .online import OnlineController
from . import dg as DG

_rng = np.random.default_rng(C.S4_SEED)
# 非线程安全：_rng与b1_step.state为全局，批量用多进程（勿用多线程并发run，B1会分叉~1kg，见边界F8）


def _reset_b1_state():
    global _rng
    _rng = np.random.default_rng(C.S4_SEED)  # 噪声流重置，保证跨次可复现
    b1_step.state = {"n": None, "d": None, "t": None, "eta": None}


def b1_step(v_ms, ws, wa):
    """B1：专利口径复现（权9四步法）。
    S1参数采集（入参即采集值）→ S2前馈（表1给n0/d0/t0初值）→
    S3在线寻优（坐标轮换爬山max专利η，GD_STEPS步封顶，模拟在线算力）→
    S4反馈锁定（传感器2%噪声，实测η低于上周期S4_TOL则回退锁定）。
    d、t 为连续物理值（专利S3在线优化三变量，不分档）。
    """
    si, sj = segment(ws, wa)
    n = C.N0_FF[si]
    d = C.D_VALS[C.D_GEAR_MAP[si]]
    t = C.T_VALS[C.T_GEAR_MAP[sj]]
    for _ in range(C.GD_STEPS):
        e0 = eta_patent(n, d, t, ws)
        moved = False
        for cn, cd, ct in ((n + C.GD_SN, d, t), (n - C.GD_SN, d, t),
                           (n, d + C.GD_SD, t), (n, d - C.GD_SD, t),
                           (n, d, t + C.GD_ST), (n, d, t - C.GD_ST)):
            if not (0.0 <= cn <= C.N_MAX
                    and C.D_MIN <= cd <= C.D_MAX
                    and C.T_MIN <= ct <= C.T_MAX):
                continue
            if eta_patent(cn, cd, ct, ws) > e0:
                n, d, t = cn, cd, ct
                moved = True
                break
        if not moved:
            break
    e_meas = eta_patent(n, d, t, ws) * (1.0 + C.S4_NOISE * _rng.standard_normal())
    s = b1_step.state
    if s["eta"] is not None and e_meas < s["eta"] * (1.0 - C.S4_TOL):
        n, d, t = s["n"], s["d"], s["t"]
    else:
        s.update(n=n, d=d, t=t, eta=e_meas)
    f, _, p_dg = total_fuel_rate(v_ms, ws, n, d, t)
    return f, p_dg


b1_step.state = {"n": None, "d": None, "t": None, "eta": None}


def run(voyage, dt_s, tables_path):
    ctl = OnlineController(tables_path)
    _reset_b1_state()  # 每个航次重置S4锁定状态
    # 多机在线状态：四策略各独立n_online，初值1（hotel必须有人供电）
    n_online = {k: 1 for k in ("B0", "B1", "B2", "ours")}
    acc = {k: {"fuel": 0.0, "dg_low": 0, "n": 0, "extra": 0.0,
               "overload": 0, "switches": 0,
               "p_dg_min": 1e18, "p_dg_max": -1e18}
           for k in ("B0", "B1", "B2", "ours")}
    t_lookup = []
    for ws, wa, vs_kn in voyage:
        v_ms = vs_kn * 1852.0 / 3600.0
        # B0：无帆（字面：不装筒。T=0，无待机Pf；P_ME全担船阻，DG只带hotel）
        R0 = hull_resistance(v_ms)
        p_me0 = R0 * v_ms / 1000.0 / C.ETA_PROP
        p_dg0 = C.P_HOTEL
        me0 = sfoc_me(p_me0 / C.P_ME_RATED) * p_me0 / 3600.0 if p_me0 > 0 else 0.0
        # B1：专利在线寻优（η目标，S2+S3+S4）
        f1, p_dg1 = b1_step(v_ms, ws, wa)
        dg1_single = sfoc_dg(p_dg1 / C.P_DG_RATED) * p_dg1 / 3600.0 if p_dg1 > 0 else 0.0
        me1 = f1 - dg1_single
        # B2：常规（帆满转速、档位跟段、被动响应）
        si, sj = segment(ws, wa)
        f2, p_me2, p_dg2 = total_fuel_rate(v_ms, ws, C.N_MAX,
                                       C.D_VALS[C.D_GEAR_MAP[si]],
                                       C.T_VALS[C.T_GEAR_MAP[sj]])
        me2 = sfoc_me(p_me2 / C.P_ME_RATED) * p_me2 / 3600.0 if p_me2 > 0 else 0.0
        # ours：查表（表给n*，油耗按物理重算，与旧口径一致；表pme仅参考不进油耗）
        n_tbl, _pme_tbl, di, ti, dt = ctl.step(ws, wa, vs_kn, return_pme=False)
        t_lookup.append(dt)
        f, p_me_r, p_dg = total_fuel_rate(v_ms, ws, n_tbl, C.D_VALS[di], C.T_VALS[ti])
        me = sfoc_me(p_me_r / C.P_ME_RATED) * p_me_r / 3600.0 if p_me_r > 0 else 0.0
        # 多机在线执行：各策略独立dg_opt，附加计入本步
        for k, me_rate, pd in (("B0", me0, p_dg0),
                               ("B1", me1, p_dg1),
                               ("B2", me2, p_dg2),
                               ("ours", me, p_dg)):
            n_new, extra = DG.dg_opt(pd, n_online[k], dt_s)
            dg_rate = DG.fuel_dg_rate(pd, n_new)
            if dg_rate == float("inf"):
                # 超N_DG_MAX×P_RATED（本批最大~2040kW不会触发）：记过载，
                # 用单台外推保有限值，不污染总量可比性
                acc[k]["overload"] += 1
                dg_rate = sfoc_dg(pd / C.P_DG_RATED) * pd / 3600.0 if pd > 0 else 0.0
                extra = 0.0
            else:
                # dg_opt无可行台数分支（返回MAX,0）同样记过载一步
                if not any(DG.feasible_n(pd, nn)
                           for nn in range(1, DG.N_DG_MAX + 1)) and pd > 0:
                    acc[k]["overload"] += 1
            if n_new != n_online[k]:
                acc[k]["switches"] += 1
            n_online[k] = n_new
            acc[k]["fuel"] += me_rate * dt_s + dg_rate * dt_s + extra
            acc[k]["extra"] += extra
            acc[k]["n"] += 1
            acc[k]["p_dg_min"] = min(acc[k]["p_dg_min"], float(pd))
            acc[k]["p_dg_max"] = max(acc[k]["p_dg_max"], float(pd))
            if pd / C.P_DG_RATED < C.DG_LOW_EFF:
                acc[k]["dg_low"] += 1
    rep = {}
    for k in acc:
        rep[k] = {"fuel_t": acc[k]["fuel"] / 1e6,
                  "save_vs_B0": 1.0 - acc[k]["fuel"] / acc["B0"]["fuel"],
                  "dg_low_frac": acc[k]["dg_low"] / acc[k]["n"],
                  "extra_g": acc[k]["extra"],
                  "n_switches": acc[k]["switches"],
                  "n_overload": acc[k]["overload"],
                  "p_dg_min": acc[k]["p_dg_min"],
                  "p_dg_max": acc[k]["p_dg_max"]}
    rep["lookup_ms"] = float(np.mean(t_lookup)) * 1000.0
    return rep


if __name__ == "__main__":
    from .synth import make_voyage
    voyage, dt_s = make_voyage()
    rep = run(voyage, dt_s, "tables.npz")
    for k in ("B0", "B1", "B2", "ours"):
        r = rep[k]
        print(f"{k}: fuel={r['fuel_t']:.3f}t "
              f"save={r['save_vs_B0'] * 100:.2f}% "
              f"dg_low={r['dg_low_frac'] * 100:.1f}%")
    print(f"lookup mean: {rep['lookup_ms']:.3f} ms")
