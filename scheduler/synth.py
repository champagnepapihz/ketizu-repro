"""合成航次：覆盖 低/中/高风速 x 正向/中等/侧向风 x 典型船速，并含段间切换。
返回每步 (相对风速ws, 来流角wa, 船速vs_kn)。
"""
import numpy as np


def make_voyage(dt_s=60.0):
    segs = [  # (ws, wa_deg, vs_kn, 步数)，船速不超表包线
        (6.0, 20.0, 12.0, 40),    # 低风速正向
        (6.0, 100.0, 12.0, 40),   # 低风速侧向
        (11.0, 60.0, 13.0, 40),   # 中风速中等
        (11.0, 120.0, 13.0, 40),  # 中风速侧向
        (17.0, 45.0, 13.0, 40),   # 高风速中等
        (17.0, 10.0, 13.0, 40),   # 高风速正向
        (5.0, 150.0, 12.0, 40),   # 回落：低风速侧向
    ]
    out = []
    rng = np.random.default_rng(0)
    for ws, wa, vs, n in segs:
        out.append(np.column_stack([
            ws + rng.normal(0, 0.5, n),
            wa + rng.normal(0, 3.0, n),
            np.full(n, vs),
        ]))
    return np.vstack(out), dt_s
