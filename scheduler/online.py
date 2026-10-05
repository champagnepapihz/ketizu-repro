"""在线：查表 + 插值给设定值；换档过三关（变段确认、去抖、最短驻留）。
返回每步 (n*, P_ME*, d, t, 耗时s)。

域声明：有效域 ws∈[4.0,20.0]m/s、wa∈[0,180]deg、vs∈[10.0,13.0]kn（见config WS/WA/VS_GRID）；
域外输入钳制到包线（_clamp）并计oob_count+1，钳制值非真值，论文引用域外结论须声明。
非线程安全说明见simulate（_rng/b1_state全局）；本类oob_count仅计数用途。
"""
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from . import config as C
from .build_tables import segment


class OnlineController:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        pts = (z["ws"], z["wa"], z["vs"],
               np.arange(3), np.arange(3))
        self._fn = RegularGridInterpolator(pts, z["n_star"],
                                           method="linear",
                                           bounds_error=False,
                                           fill_value=None)
        self._fp = RegularGridInterpolator(pts, z["pme_star"],
                                           method="linear",
                                           bounds_error=False,
                                           fill_value=None)
        self.d, self.t = 1, 1
        self._cand = None
        self._cand_n = 0
        self._dwell = 0
        self.oob_count = 0  # 域外步计数（只计数，不改返回签名）

    def _clamp(self, ws, wa, vs):
        ws = float(np.clip(ws, C.WS_GRID[0], C.WS_GRID[-1]))
        wa = float(np.clip(wa, C.WA_GRID[0], C.WA_GRID[-1]))
        vs = float(np.clip(vs, C.VS_GRID[0], C.VS_GRID[-1]))
        return ws, wa, vs

    def step(self, ws, wa, vs_kn, return_pme=True):
        if not (np.isfinite(ws) and np.isfinite(wa) and np.isfinite(vs_kn)):
            raise ValueError(f"step非有限输入 ws={ws} wa={wa} vs={vs_kn}")
        # 域外计数（只计数不改签名；钳制值非真值）
        if (ws < C.WS_GRID[0] or ws > C.WS_GRID[-1]
                or wa < C.WA_GRID[0] or wa > C.WA_GRID[-1]
                or vs_kn < C.VS_GRID[0] or vs_kn > C.VS_GRID[-1]):
            self.oob_count += 1
        # --- 换档三关 ---
        si, sj = segment(ws, wa)
        want_d = C.D_GEAR_MAP[si]
        want_t = C.T_GEAR_MAP[sj]
        if self._dwell > 0:
            self._dwell -= 1
        elif (want_d, want_t) != (self.d, self.t):
            if self._cand == (want_d, want_t):
                self._cand_n += 1
            else:
                self._cand, self._cand_n = (want_d, want_t), 1
            if self._cand_n >= C.DEBOUNCE_N:
                self.d, self.t = want_d, want_t
                self._cand_n = 0
                self._dwell = C.DWELL_STEPS
        else:
            self._cand_n = 0
        # --- 查表 ---
        ws, wa, vs = self._clamp(ws, wa, vs_kn)
        t0 = time.perf_counter()
        n = float(self._fn([(ws, wa, vs, self.d, self.t)])[0])
        if return_pme:
            p = float(self._fp([(ws, wa, vs, self.d, self.t)])[0])
        else:
            p = 0.0  # 转正后simulate物理重算，不用表pme，跳过一半插值省时
        dt = time.perf_counter() - t0
        return max(n, 0.0), max(p, 0.0), self.d, self.t, dt
