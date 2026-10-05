"""物理模型：接口按《优化问题形式化》定义；CL/Pf 采用专利线性族原文形式。
专利：CL=CL0+k1*n+k2/d+k3*cosθ；Pf=c1*n^2+c2/d+c3*sinθ+c4；
      η=CL*ρ*A*vr^3/(Pm+Pf)，A=D×H。
系数值为占位（TODO(组会)：风洞标定）；单调性与专利一致。
所有功率单位 kW，油耗 g/kWh，燃油 g/s 需自行乘时间。
d、t 一律物理值（m、deg），调用方负责档位→物理值转换。
"""
import numpy as np
from . import config as C


def _area():
    return C.N_SAILS * C.D_ROTOR * C.H_ROTOR


def sail_thrust(n_rpm, d, t, vr):
    """帆推力 N。专利CL形式：CL = CL0 + K1*n + K2/d + K3*cosθ（θ取deg转弧度）。
    d、t 为物理值（m、deg）。"""
    vr = max(vr, 0.5)
    cl = _cl(n_rpm, d, t)
    return 0.5 * C.RHO_AIR * _area() * cl * vr ** 2


def motor_power(n_rpm):
    """帆电机总耗电 kW。专利Pm(n)仅转速函数（电参数估算，占位取线性族）。"""
    return C.N_SAILS * C.M1 * n_rpm


def friction_power(n_rpm, d, t):
    """系统摩擦功率 kW（全船）。专利Pf形式：
    Pf = N*(C1*n^2 + C2/d + C3*sinθ + C4)，θ取deg转弧度。系数占位。"""
    import math
    if not (bool(np.all(np.isfinite(n_rpm))) and bool(np.all(np.isfinite(d))) and bool(np.all(np.isfinite(t)))):
        raise ValueError(f"friction_power非有限输入 n={n_rpm} d={d} t={t}")
    if float(d) <= 0:
        raise ValueError(f"friction_power要求d>0，得d={d}")
    pf_1 = (C.C1 * n_rpm ** 2 + C.C2 / float(d)
            + C.C3 * math.sin(math.radians(float(t))) + C.C4)
    return C.N_SAILS * pf_1


def sfoc_me(load):
    """主机 SFOC g/kWh。MAN拟合真值：U 形二次曲线。"""
    return C.SFOC_ME_MIN + C.A_ME * (load - C.X_ME_OPT) ** 2


def sfoc_dg(load):
    """柴油发电机 SFOC g/kWh。MAN拟合真值：U 形，低负荷惩罚更陡。"""
    return C.SFOC_DG_MIN + C.A_DG * (load - C.X_DG_OPT) ** 2


def hull_resistance(v_ms):
    """船阻力 N。占位：平方律。"""
    return C.C_RES * v_ms ** 2


def total_fuel_rate(v_ms, vr, n_rpm, d, t):
    """给定工况与转速，返回 (总油耗率 g/s, 主机功率 kW, 发电功率 kW)。
    d、t 为物理值（m、deg）。

    P_ME 由推力平衡决定（船速不掉）：P_ME = max(0, (R - T) * V / eta)。
    P_DG 由电力平衡决定：P_DG = P_hotel + Pm(n) + Pf(n,d,t)。
    """
    T = sail_thrust(n_rpm, d, t, vr)
    R = hull_resistance(v_ms)
    p_me = max(0.0, (R - T) * v_ms / 1000.0 / C.ETA_PROP)
    p_dg = C.P_HOTEL + motor_power(n_rpm) + friction_power(n_rpm, d, t)
    f = (sfoc_me(p_me / C.P_ME_RATED) * p_me
         + sfoc_dg(p_dg / C.P_DG_RATED) * p_dg) / 3600.0
    return f, p_me, p_dg


def feasible(p_me, p_dg):
    """硬约束：主机/发电机不超额定（形式化文档约束3）。"""
    return p_me <= C.P_ME_RATED and p_dg <= C.P_DG_RATED


def eta_patent(n_rpm, d, t, vr):
    """专利η口径（B1优化目标）：η = CL*ρ*A*vr^3/(Pm+Pf)，A=D×H单筒迎风面积。
    d、t 为物理值（m、deg）。专利S3最大化η求(n,d,θ)，S4实测对比锁定。"""
    vr = max(vr, 0.5)
    cl = _cl(n_rpm, d, t)
    num = cl * C.RHO_AIR * (C.D_ROTOR * C.H_ROTOR) * vr ** 3
    den = motor_power(n_rpm) * 1000.0 + friction_power(n_rpm, d, t) * 1000.0
    return num / max(den, 1e-9)


def _cl(n_rpm, d, t):
    """专利CL形式抽取，供 eta_patent 复用。"""
    import math
    if not (bool(np.all(np.isfinite(n_rpm))) and bool(np.all(np.isfinite(d))) and bool(np.all(np.isfinite(t)))):
        raise ValueError(f"_cl非有限输入 n={n_rpm} d={d} t={t}")
    if float(d) <= 0:
        raise ValueError(f"_cl要求d>0，得d={d}")
    return (C.CL0 + C.K1 * float(n_rpm)
            + C.K2 / float(d) + C.K3 * math.cos(math.radians(float(t))))
