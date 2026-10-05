"""离线穷举建精调表：给定档位下对转速网格穷举取最小（全局，不陷局部）。
档位表照搬专利（config 中映射），本脚本只存精调表。
输出 tables.npz：n_star[ws, wa, vs, d, t], pme_star, grids。

多机转正版（2026-10-02，dg转正实验）：
- 可行域改多机版：存在开1..N_DG_MAX台使P_DG可行即算可行（用dg.feasible_n）。
  即 P_DG <= N_DG_MAX×P_RATED 即可，不再按单台1500卡。
- 目标函数改多机版：ME油耗 + dg.fuel_dg_rate(p_dg, steady_n(p_dg))，
  steady_n为运行油耗最小台数（不计启动/切换罚，见dg.py）。
  P_DG<=1500时与单台等效逐克一致（向后兼容）；P_DG>1500时用真实多机油耗。
- 表结构不动：仍只存n_star/pme_star，开机数由P_DG派生（在线由dg_opt给出）。
- 新增参数无（全复用dg.py占位参数N_DG_MAX/X_DG_MIN/K_START/FUEL_SWITCH，TODO见dg.py）。
"""
import numpy as np
from . import config as C
from .models import total_fuel_rate, sfoc_me
from . import dg as DG


def segment(ws, wa):
    i = int(np.digitize(ws, C.WS_BANDS)) - 1
    j = int(np.digitize(wa, C.WA_BANDS)) - 1
    return min(max(i, 0), 2), min(max(j, 0), 2)


def feasible_multi(p_me, p_dg):
    """多机可行：主机不超额定，且存在1..N_DG_MAX台使P_DG可行。

    用dg.feasible_n逐台判定（均分假设，占位，TODO见dg.py）。
    表结构不动，开机数由P_DG派生，这里只判存在性。
    """
    if p_me > C.P_ME_RATED + 1e-9:
        return False
    for n in range(1, DG.N_DG_MAX + 1):
        if DG.feasible_n(p_dg, n):
            return True
    return False


def total_fuel_rate_multi(v_ms, ws, n_rpm, d, t):
    """多机口径总油耗率（g/s）+分量，用于建表寻优。

    ME部分沿用models单机公式；DG部分用dg.fuel_dg_rate(p_dg, steady_n(p_dg))。
    P_DG<=1500时steady_n恒1，与单台等效一致；P_DG>N_DG_MAX×P_RATED时返回inf（不可行）。
    """
    f_single, p_me, p_dg = total_fuel_rate(v_ms, ws, n_rpm, d, t)
    n_dg = DG.steady_n(p_dg)
    dg_rate = DG.fuel_dg_rate(p_dg, n_dg)
    if dg_rate == float("inf"):
        return float("inf"), p_me, p_dg
    me_rate = sfoc_me(p_me / C.P_ME_RATED) * p_me / 3600.0 if p_me > 0 else 0.0
    return me_rate + dg_rate, p_me, p_dg


def build(out_path):
    nws, nwa, nvs = len(C.WS_GRID), len(C.WA_GRID), len(C.VS_GRID)
    n_star = np.zeros((nws, nwa, nvs, 3, 3))
    pme_star = np.zeros_like(n_star)
    n_infeasible = 0
    for a, ws in enumerate(C.WS_GRID):
        for b, wa in enumerate(C.WA_GRID):
            for c, vs_kn in enumerate(C.VS_GRID):
                v_ms = vs_kn * 1852.0 / 3600.0
                for d_idx in range(3):
                    for t_idx in range(3):
                        d, t = C.D_VALS[d_idx], C.T_VALS[t_idx]
                        best, bn, bp = 1e18, 0.0, 0.0
                        fb_pdg, fbn, fbp = 1e18, 0.0, 0.0
                        for n in C.N_GRID:
                            f, p_me, p_dg = total_fuel_rate_multi(v_ms, ws, n, d, t)
                            if feasible_multi(p_me, p_dg) and f < best:
                                best, bn, bp = f, n, p_me
                            if p_dg < fb_pdg:
                                fb_pdg, fbn, fbp = p_dg, n, p_me
                        if best > 1e17:  # 该格无可行点：退为 P_DG 最小（DG主因时不加剧过载）
                            bn, bp = fbn, fbp
                            n_infeasible += 1
                        n_star[a, b, c, d_idx, t_idx] = bn
                        pme_star[a, b, c, d_idx, t_idx] = bp
    np.savez(out_path, n_star=n_star, pme_star=pme_star,
             ws=C.WS_GRID, wa=C.WA_GRID, vs=C.VS_GRID)
    print(f"built {out_path}: shape={n_star.shape} "
          f"infeasible_cells={n_infeasible}")


if __name__ == "__main__":
    build("tables.npz")
