"""多台发电机投切（多机已接入，转正后主流程在线执行）。

地位：多机转正已接入（build_tables 可行域＋目标函数、simulate 在线执行），
后处理口径作废；本文件提供 steady_n/dg_opt/fuel_dg_rate/feasible_n 供主流程调用。

模型（与 tex “离线算好、在线照表执行”对应）：
- 单台额定复用 config.P_DG_RATED（当前 1500kW，占位）；
  总容量 = n_online × P_DG_RATED；多台均分负荷。
- 运行油耗沿用 models.sfoc_dg 的 U 形曲线，只是负荷率按
  x = p_dg / n / P_DG_RATED 计算，总油耗率 = SFOC(x)×p_dg/3600。
- 离线：对给定 p_dg 枚举 n=1..N_DG_MAX，取满足容量+最低负荷率的
  运行油耗最小者为稳态最优（即离线表值，在线照表执行）。
- 在线：在稳态最优基础上加启动分摊与切换罚，选单步总成本最小，
  起到换档去抖/驻留类似的防抖作用（类比 online.py 三关思想）。

全部新增参数均为占位，已标 TODO(组会)，见下。
TODO(组会)：单台额定/台数/最低负荷率/启动油耗系数/切换罚均需实船数据标定。
"""
from . import config as C
from .models import sfoc_dg

# ---------------- 占位参数（TODO(组会)：实船标定） ----------------
N_DG_MAX = 3            # 最大可投运台数（占位，TODO(组会)：按电站配置定1..3/1..4）
X_DG_MIN = 0.30         # 单台最低负荷率下限，低于此且可减机时必须停一台（占位，TODO(组会)）
K_START_FRAC = 0.005    # 启动油耗系数：单台额定×K_START_FRAC×1h折能量再×SFOC（占位，TODO(组会)）
FUEL_SWITCH_G = 500.0   # 每次切换固定罚（g/次，含离合/同步折耗，占位，TODO(组会)）


def fuel_start_g(n_add=1):
    """启动附加油耗（g）。占位口径：P_RATED×K_START_FRAC×1h能量 × SFOC_DG_MIN。

    TODO(组会)：换实测启动油耗（g/次·台）。
    """
    return max(0, int(n_add)) * C.P_DG_RATED * K_START_FRAC * C.SFOC_DG_MIN


def feasible_n(p_dg, n):
    """容量可行：p_dg <= n×P_RATED（均分假设）。"""
    return 1 <= n <= N_DG_MAX and p_dg <= n * C.P_DG_RATED + 1e-9


def fuel_dg_rate(p_dg, n):
    """n 台均分时的柴油运行油耗率（g/s）。不可行返回 inf。

    p_dg<=0 时返回 0（停船/无电负荷边界，占位处理）。
    """
    if p_dg <= 0:
        return 0.0
    if not feasible_n(p_dg, n):
        return float("inf")
    x = p_dg / n / C.P_DG_RATED
    return sfoc_dg(x) * p_dg / 3600.0


def _candidates(p_dg):
    """满足容量+最低负荷率的候选台数。

    规则：先取容量可行集，最小可行台数 n_min 恒保留（hotel 必须有人供电，
    即使 x<X_DG_MIN 也只能开着）；n>n_min 且 x<X_DG_MIN 的候选剔除
    （“低于30%必须停一台，防低效”）。
    若容量全不可行（p_dg>MAX×P_RATED），返回 []，由调用方记过载。
    """
    feas = [n for n in range(1, N_DG_MAX + 1) if feasible_n(p_dg, n)]
    if not feas:
        return []
    n_min = min(feas)
    out = [n for n in feas
           if n == n_min or p_dg / n / C.P_DG_RATED >= X_DG_MIN]
    return out if out else [n_min]


def steady_n(p_dg):
    """稳态（离线表值）：运行油耗最小的台数，不计启动/切换罚。

    即“离线枚举开1..3台选总油耗最小”中不计罚项的版本；
    在线照表执行时以此为基准，再由 dg_opt 加罚做防抖。
    无可行台数时返回 N_DG_MAX（调用方记过载）。
    """
    cands = _candidates(p_dg)
    if not cands:
        return N_DG_MAX
    return min(cands, key=lambda n: (fuel_dg_rate(p_dg, n), n))


def dg_opt(p_dg, n_online=1, dt_s=None):
    """单步投切决策（在线用）。

    Args:
        p_dg: 本步电力需求 kW（simulate 既有 P_DG 口径：hotel+Pm+Pf）。
        n_online: 上一步在线台数。
        dt_s: 本步时长 s，默认 config.CONTROL_DT_S（占位60s）。

    Returns:
        (n_new, fuel_extra_g)：n_new 为本步投运台数；
        fuel_extra_g 为本步附加罚（启动+切换，g），运行油耗另由
        fuel_dg_rate(p_dg, n_new)×dt_s 计算，不混在一起。
        若 p_dg 容量全不可行，返回 (N_DG_MAX, 0.0)，调用方记过载一步。

    总成本口径（选最小者）：
        total = rate(n)×dt_s + max(0,n-n_online)×START + (0若n==n_online else SWITCH)。
    平票取小 n（少开一台）。
    """
    if dt_s is None:
        dt_s = C.CONTROL_DT_S
    try:
        n_online = int(n_online)
    except Exception:
        n_online = 1
    if n_online not in range(1, N_DG_MAX + 1):
        n_online = 1
    cands = _candidates(p_dg)
    if not cands:
        return N_DG_MAX, 0.0
    best, best_cost, best_extra = None, None, 0.0
    for n in cands:
        rate = fuel_dg_rate(p_dg, n)
        start = fuel_start_g(n - n_online) if n > n_online else 0.0
        switch = 0.0 if n == n_online else FUEL_SWITCH_G
        total = rate * dt_s + start + switch
        key = (total, n)
        if best_cost is None or key < best_cost:
            best, best_cost = n, key
            best_extra = start + switch
    return best, best_extra


def apply_trace(p_list, n0=1, dt_s=None):
    """整条 P_DG 迹线的投切回放（供临时验证脚本与未来在线监督层复用）。

    Returns:
        dict(n_trace, fuel_op_g, fuel_extra_g, fuel_total_g, n_switches, n_overload)：
        fuel_op 为各步运行油耗之和，fuel_extra 为启动+切换罚之和。
    TODO(组会)：未来电侧监督降载层可直接复用本函数做“投切+降载”联合回放。
    """
    if dt_s is None:
        dt_s = C.CONTROL_DT_S
    n_trace = []
    fuel_op, fuel_extra, overload = 0.0, 0.0, 0
    n_cur = n0
    for p in p_list:
        if not _candidates(p):
            overload += 1
            n_trace.append(N_DG_MAX)
            fuel_op += fuel_dg_rate(p, N_DG_MAX) * dt_s  # inf，调用方按需处理
            continue
        n_new, extra = dg_opt(p, n_cur, dt_s)
        n_trace.append(n_new)
        fuel_op += fuel_dg_rate(p, n_new) * dt_s
        fuel_extra += extra
        n_cur = n_new
    switches = sum(1 for a, b in zip([n0] + n_trace[:-1], n_trace) if a != b)
    return {"n_trace": n_trace, "fuel_op_g": fuel_op,
            "fuel_extra_g": fuel_extra, "fuel_total_g": fuel_op + fuel_extra,
            "n_switches": switches, "n_overload": overload}
