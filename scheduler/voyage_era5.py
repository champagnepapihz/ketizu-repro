"""ERA5 真实风场航次接线：合成航次 -> 再分析风场驱动。

地位：scheduler/ 下唯一允许的新增文件（铁律：不碰现有文件与 tex）。
接口与 synth.make_voyage 同格式：返回 (voyage[Nx3], dt_s)，
voyage 每行 = (ws相对风速 m/s, wa来流角 deg, vs_kn船速 kn)，dt_s=3600.0（逐小时）。

定义约定（必读，与 config.WA_BANDS 对齐）：
  * 真风 (u, v)：ERA5 10m u/v 分量，空气相对地面的运动矢量，单位 m/s。
    u=向东分量，v=向北分量。风速=hypot(u,v)；风向（去向）=atan2(u,v)。
  * 船向 hdg：船艏方位角，deg，0=正北，顺时针（0=N,90=E），与航海惯例一致。
  * 船速 Vs：m/s（输入 kn 需 x1852/3600 换算）。
  * 表观（相对）风矢量 r = 真风 - 船速矢量（专利 S1 原文"vr 为相对风速，
    由船速与真实风速矢量合成"）：
      su = Vs*sin(hdg), sv = Vs*cos(hdg)   # 船速矢量（东，北）
      ru = uz - su, rv = vz - sv           # 表观风矢量（东，北）
      ws = hypot(ru, rv)
  * 来流角 wa：表观风矢量 r 与船艏单位矢量 f=(sin hdg, cos hdg) 的夹角，
    wa = arccos((r.f)/|r|)，范围 [0,180]：
      wa=0   = 顺风推船（空气与船同向运动，风从艉来）-> config 正向段 0-30
      wa=180 = 顶风（空气与船反向运动，风从艏来）  -> config 侧向段 90-181
    注意：这是"风矢量夹角"，不是"风从哪来"的气象来向；与 synth.py 语义一致
    （synth 中 wa=20 正向小角度 = 顺风，wa=100+ 侧向大角度 = 偏顶风）。
    ws≈0 时 wa 无定义，本函数返回 0.0（ documented edge ）。

高度外推（占位假设，论文需细化）：
  筒体高度风 u(z) = u10 * (z/10)^alpha，alpha=1/7 占位，
  z 默认取 config.H_ROTOR/2（筒高中点，当前 12m）。方向不变，只缩放大小。
  再分析是 10m 风，不是船载实测，不是筒体高度实测，此处外推为占位口径。

插值：最近邻（TODO(组会)：换双线性；航线跨 0-360 经度拼接处需 wrap 处理）。

时间->位置映射：船 t=0 从航路点 0 出发，以 Vs 恒速沿大圆/恒向分段推进，
  每小时一个点（dt_s=3600）。若 nc 时间步数超过单程时长，默认停在终点
  （loop=False）；置 loop=True 则循环绕航（全年 8760h 多航次占位口径）。
"""

import numpy as np

try:
    from . import config as C
except ImportError:  # 允许 temp 脚本直接 sys.path 挂 scheduler 目录时导入
    import config as C  # type: ignore

DT_S = 3600.0
ALPHA_DEFAULT = 1.0 / 7.0  # 幂律指数占位，论文需细化

# Ras Tanura（沙特）-> 舟山 VLCC 线默认航路点（lat, lon），共 18 点。
# 坐标为公开航线示意（占位精度，论文需用 ECDIS/实船航迹细化）。
DEFAULT_WAYPOINTS = [
    (26.65, 50.17),    # 1 Ras Tanura 锚地
    (26.30, 56.50),    # 2 霍尔木兹入口
    (25.20, 58.00),    # 3 阿曼湾
    (22.00, 62.00),    # 4 阿拉伯海西
    (18.00, 65.00),    # 5 阿拉伯海中
    (12.00, 66.00),    # 6 阿拉伯海南
    (5.80, 81.50),     # 7 斯里兰卡南
    (5.50, 84.50),     # 8 斯里兰卡东
    (6.50, 92.00),     # 9 尼科巴西
    (5.50, 97.50),     # 10 马六甲入口
    (4.00, 100.00),    # 11 马六甲中（槟城）
    (1.20, 103.80),    # 12 新加坡海峡
    (4.00, 109.50),    # 13 南海南
    (10.00, 113.50),   # 14 南海中
    (16.00, 116.00),   # 15 南海北
    (21.50, 118.50),   # 16 台湾海峡南
    (26.00, 122.00),   # 17 东海
    (29.90, 122.20),   # 18 舟山
]


def _height_factor(z_m=None, alpha=ALPHA_DEFAULT):
    """幂律外推因子 (z/10)^alpha。z 默认 H_ROTOR/2。占位假设，论文需细化。"""
    if z_m is None:
        z_m = float(C.H_ROTOR) / 2.0
    return float((float(z_m) / 10.0) ** float(alpha))


def true_to_apparent(u10, v10, ship_heading_deg, ship_speed_ms,
                     z_m=None, alpha=ALPHA_DEFAULT):
    """真风 -> 表观风矢量合成。

    Args:
        u10, v10: 10m 真风分量 m/s（标量或 ndarray，可广播）。
        ship_heading_deg: 船艏方位角 deg（0=N 顺时针）。
        ship_speed_ms: 船速 m/s（标量或可广播数组）。
        z_m: 外推高度 m，默认 H_ROTOR/2（占位假设，论文需细化）。
        alpha: 幂律指数，默认 1/7（占位假设，论文需细化）。

    Returns:
        (ws, wa)：表观风速 m/s，来流角 deg[0,180]（0=顺风推船，180=顶风）。
    """
    if not (bool(np.all(np.isfinite(np.asarray(u10, dtype=float))))
            and bool(np.all(np.isfinite(np.asarray(v10, dtype=float))))
            and bool(np.all(np.isfinite(np.asarray(ship_heading_deg, dtype=float))))
            and bool(np.all(np.isfinite(np.asarray(ship_speed_ms, dtype=float))))):
        raise ValueError("true_to_apparent非有限输入")
    factor = _height_factor(z_m, alpha)
    u = np.asarray(u10, dtype=float) * factor
    v = np.asarray(v10, dtype=float) * factor
    hdg = np.radians(np.asarray(ship_heading_deg, dtype=float))
    vs = np.asarray(ship_speed_ms, dtype=float)
    su = vs * np.sin(hdg)
    sv = vs * np.cos(hdg)
    ru = u - su
    rv = v - sv
    ws = np.hypot(ru, rv)
    fx = np.sin(hdg)
    fy = np.cos(hdg)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_wa = (ru * fx + rv * fy) / np.maximum(ws, 1e-12)
    cos_wa = np.clip(cos_wa, -1.0, 1.0)
    wa = np.degrees(np.arccos(cos_wa))
    # ws≈0 奇异：夹角无定义，置 0（顺风占位），避免 NaN 污染仿真
    if np.ndim(wa) == 0:
        return float(ws), float(0.0 if float(ws) < 1e-9 else float(wa))
    wa = np.where(ws < 1e-9, 0.0, wa)
    return ws, wa


def _pick_var(variables, candidates):
    for k in candidates:
        if k in variables:
            return k
    return None


def read_era5_nc(nc_path):
    """读 ERA5 NetCDF（CDS reanalysis-era5-single-levels 口径）。

    支持变量名别名：time/valid_time；latitude/lat；longitude/lon；
    u10/10m_u_component_of_wind；v10/10m_v_component_of_wind。
    维度自动转置为 (time, lat, lon)。若文件为 .npz（含 time/lat/lon/u10/v10
    或 latitude/longitude/u10/v10 键）则直接读取——用于真实 nc 未到之前的
    假数据占位（见 ERA5接线说明.md）。

    Returns:
        dict(time, lat, lon, u10, v10, path, source)：u10/v10 形状 (nt, nlat, nlon)。
    """
    import os
    path = str(nc_path)
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        keys = set(z.files)
        lat_k = _pick_var(keys, ["lat", "latitude", "lats"])
        lon_k = _pick_var(keys, ["lon", "longitude", "lons"])
        u_k = _pick_var(keys, ["u10", "10m_u_component_of_wind", "u", "U10"])
        v_k = _pick_var(keys, ["v10", "10m_v_component_of_wind", "v", "V10"])
        t_k = _pick_var(keys, ["time", "valid_time", "times"])
        assert lat_k and lon_k and u_k and v_k and t_k, \
            f"npz 缺键 files={sorted(keys)}"
        return {"time": np.asarray(z[t_k]), "lat": np.asarray(z[lat_k]),
                "lon": np.asarray(z[lon_k]),
                "u10": np.asarray(z[u_k], dtype=float),
                "v10": np.asarray(z[v_k], dtype=float),
                "path": path, "source": "npz-fallback"}

    # .nc：优先 netCDF4，其次 scipy.io.netcdf（本机无 netCDF4/xarray 时走后者）
    try:
        import netCDF4  # type: ignore
        ds = netCDF4.Dataset(path, "r")
        varnames = set(ds.variables.keys())

        def _arr(names):
            k = _pick_var(varnames, names)
            assert k is not None, f"nc 缺变量 {names}，现有 {sorted(varnames)}"
            v = ds.variables[k]
            dims = getattr(v, "dimensions", ())
            a = np.asarray(v[:], dtype=float)
            return a, dims, k

        t, t_dims, _ = _arr(["time", "valid_time"])
        lat, _, _ = _arr(["latitude", "lat"])
        lon, _, _ = _arr(["longitude", "lon"])
        u, u_dims, _ = _arr(["u10", "10m_u_component_of_wind", "u"])
        vv, v_dims, _ = _arr(["v10", "10m_v_component_of_wind", "v"])
        ds.close()
        u = _to_tll(u, u_dims)
        vv = _to_tll(vv, v_dims)
        return {"time": np.asarray(t), "lat": np.asarray(lat, dtype=float),
                "lon": np.asarray(lon, dtype=float),
                "u10": np.asarray(u, dtype=float),
                "v10": np.asarray(vv, dtype=float),
                "path": path, "source": "netCDF4"}
    except ImportError:
        pass

    # scipy.io.netcdf 回退（NetCDF3 classic；假 nc 即按此格式自造）
    from scipy.io import netcdf_file
    f = netcdf_file(path, "r", maskandscale=True)
    try:
        varnames = set(f.variables.keys())

        def _get(names):
            k = _pick_var(varnames, names)
            assert k is not None, f"nc 缺变量 {names}，现有 {sorted(varnames)}"
            v = f.variables[k]
            dims = getattr(v, "dimensions", ())
            a = np.array(v.data, dtype=float, copy=True)
            return a, dims

        t, _ = _get(["time", "valid_time"])
        lat, _ = _get(["latitude", "lat"])
        lon, _ = _get(["longitude", "lon"])
        u, u_dims = _get(["u10", "10m_u_component_of_wind", "u"])
        vv, v_dims = _get(["v10", "10m_v_component_of_wind", "v"])
        u = _to_tll(u, u_dims)
        vv = _to_tll(vv, v_dims)
        return {"time": np.asarray(t), "lat": np.asarray(lat, dtype=float),
                "lon": np.asarray(lon, dtype=float),
                "u10": np.asarray(u, dtype=float),
                "v10": np.asarray(vv, dtype=float),
                "path": path, "source": "scipy.io.netcdf"}
    finally:
        try:
            f.close()
        except Exception:
            pass


def _to_tll(a, dims):
    """将风场数组整理为 (time, lat, lon)。

    3D 直接按维度名转置；4D（如含 expver）取平均并注明（占位口径）。
    dims 为空（假 nc 无维度名信息）时假设已经是 (time, lat, lon)。
    """
    a = np.asarray(a)
    if a.ndim == 3:
        if not dims:
            return a
        order = _axis_order(dims)
        if order is not None:
            return np.transpose(a, order)
        return a
    if a.ndim == 4:
        # 占位：expver 维（常为 2）取平均；TODO(组会)：真数据到后按 CDS 文档取 expver=1
        if not dims:
            return np.mean(a, axis=1)
        names = [str(d).lower() for d in dims]
        # 找出 lat/lon/time 轴，剩下一轴视为 expver 求平均
        ti = _find(names, ["time", "valid_time"])
        yi = _find(names, ["latitude", "lat"])
        xi = _find(names, ["longitude", "lon"])
        others = [i for i in range(4) if i not in (ti, yi, xi)]
        b = np.mean(a, axis=others[0]) if others else a.squeeze()
        # b 现在 3D，其轴顺序为原 dims 去掉 expver 后的顺序，转为 (t,y,x)
        kept = [d for i, d in enumerate(names) if i != (others[0] if others else -1)]
        order = _axis_order(kept)
        if order is not None:
            b = np.transpose(b, order)
        return b
    if a.ndim == 2:  # 单时次 (lat, lon) -> (1, lat, lon)
        return a[np.newaxis, :, :]
    raise ValueError(f"风场维度异常 ndim={a.ndim} dims={dims}")


def _find(names, cands):
    for c in cands:
        if c in names:
            return names.index(c)
    return -1


def _axis_order(dims):
    names = [str(d).lower() for d in dims]
    ti = _find(names, ["time", "valid_time"])
    yi = _find(names, ["latitude", "lat"])
    xi = _find(names, ["longitude", "lon"])
    if ti < 0 or yi < 0 or xi < 0:
        return None
    # 当前轴 0/1/2 分别对应 dims[0]/dims[1]/dims[2]，目标顺序 (ti, yi, xi)
    want = [ti, yi, xi]
    # 求转置置换 p 使 new[p..]=old：p = argsort? 直接构造
    p = [0, 0, 0]
    for new_pos, old_pos in enumerate(want):
        p[new_pos] = old_pos
    # p 是"新轴取自旧轴"索引，np.transpose 用此语义
    if sorted(p) != [0, 1, 2]:
        return None
    order = p
    return order


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """初方位角 deg（0=N 顺时针）。"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def make_voyage_era5(nc_path, waypoints=None, vs_kn=12.5,
                     alpha=ALPHA_DEFAULT, z_m=None, loop=False):
    """ERA5 航次构造：与 synth.make_voyage 同格式。

    Args:
        nc_path: ERA5 nc（或 .npz 占位）路径。
        waypoints: [(lat, lon), ...] 航路点，默认 Ras Tanura->舟山 18 点。
        vs_kn: 恒速 kn（默认 12.5）。
        alpha, z_m: 高度外推参数（占位，论文需细化）。
        loop: False=超单程后停终点；True=循环绕航（全年多航次占位）。

    Returns:
        (voyage[Nx3], dt_s)：voyage 列=(ws, wa, vs_kn)，dt_s=3600.0。
        N = nc 时间步数（逐小时）。
    """
    if not (np.isfinite(vs_kn) and np.isfinite(alpha)):
        raise ValueError(f"make_voyage_era5非有限输入 vs_kn={vs_kn} alpha={alpha}")
    data = read_era5_nc(nc_path)
    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    u10 = np.asarray(data["u10"], dtype=float)
    v10 = np.asarray(data["v10"], dtype=float)
    nt = int(np.asarray(data["time"]).shape[0])
    assert u10.shape[0] == nt and v10.shape[0] == nt, "时间维不一致"

    wps = list(DEFAULT_WAYPOINTS if waypoints is None else waypoints)
    assert len(wps) >= 2, "至少 2 个航路点"
    wps = [(float(la), float(lo)) for la, lo in wps]
    if not all(np.isfinite(la) and np.isfinite(lo) for la, lo in wps):
        raise ValueError("make_voyage_era5航路点非有限输入")
    vs_ms = float(vs_kn) * 1852.0 / 3600.0

    # 分段长度/方位（恒速恒向分段）
    leg_len = np.array([_haversine_m(wps[i][0], wps[i][1], wps[i + 1][0], wps[i + 1][1])
                        for i in range(len(wps) - 1)], dtype=float)
    leg_hdg = np.array([_bearing_deg(wps[i][0], wps[i][1], wps[i + 1][0], wps[i + 1][1])
                        for i in range(len(wps) - 1)], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(leg_len)])
    total = float(cum[-1])

    out = np.zeros((nt, 3), dtype=float)
    for t in range(nt):
        s = vs_ms * (t * DT_S)  # t=0 在起点
        if s >= total:
            if loop and total > 0:
                s = s % total
            else:
                s = total
        # 定位所在 leg：cum[k] <= s < cum[k+1]，终点取最后一段
        if s >= total:
            k = len(leg_len) - 1
            plat, plon = wps[-1]
            hdg = float(leg_hdg[-1])
        else:
            k = int(np.searchsorted(cum, s, side="right")) - 1
            k = min(max(k, 0), len(leg_len) - 1)
            frac = (s - cum[k]) / max(leg_len[k], 1e-9)
            plat = wps[k][0] + frac * (wps[k + 1][0] - wps[k][0])
            plon = wps[k][1] + frac * (wps[k + 1][1] - wps[k][1])
            hdg = float(leg_hdg[k])
        # 最近邻插值（TODO(组会)：换双线性插值）
        j = int(np.argmin(np.abs(lat - plat)))
        # 经度：ERA5 可能是 0-360，本航线 50-122E，直接最近邻即可（TODO wrap）
        i = int(np.argmin(np.abs(lon - plon)))
        uu = float(u10[t, j, i])
        vv = float(v10[t, j, i])
        ws, wa = true_to_apparent(uu, vv, hdg, vs_ms, z_m=z_m, alpha=alpha)
        out[t, 0] = float(np.clip(ws, 0.0, 60.0))  # 防脏数据；包线外 online.py 另 clamp
        out[t, 1] = float(np.clip(wa, 0.0, 180.0))
        out[t, 2] = float(vs_kn)
    return out, float(DT_S)
