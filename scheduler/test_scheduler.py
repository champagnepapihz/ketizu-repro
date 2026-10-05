"""自测试：python -m scheduler.test_scheduler（需先建表）。
断言覆盖：表形状与可行性、换档发生、查表耗时、节油排序容差、边界输入。
锚点口径：B0 fuel 8.3667t；B1 6.02% / B2 2.95%（多机转正后可行口径，单台3.70%已作废）/ ours 6.93%。
"""
import numpy as np
from . import config as C
from . import dg as DG
from . import models as M
from .online import OnlineController
from .synth import make_voyage
from .simulate import run
from .eval_predata import interp_error
from .voyage_era5 import true_to_apparent


def main():
    ctl = OnlineController("tables.npz")
    z = np.load("tables.npz")
    assert z["n_star"].shape == (len(C.WS_GRID), len(C.WA_GRID),
                                 len(C.VS_GRID), 3, 3), "表形状"
    assert bool((z["n_star"] >= 0).all()), "转速非负"
    assert bool((z["pme_star"] <= C.P_ME_RATED + 1e-9).all()), "P_ME不超额定"

    # 边界输入不崩
    for ws, wa, vs in [(-5, -10, 5), (0, 0, 10), (30, 200, 20), (11, 90, 13)]:
        n, p, d, t, dt = ctl.step(ws, wa, vs)
        assert 0 <= n <= C.N_MAX and p >= 0 and d in (0, 1, 2), "边界"

    # 航次中发生换档（合成航次含段间切换）
    ctl2 = OnlineController("tables.npz")
    seen = set()
    voyage, _ = make_voyage()
    for ws, wa, vs in voyage:
        _, _, d, t, _ = ctl2.step(ws, wa, vs)
        seen.add((d, t))
    assert len(seen) > 1, "换档未发生"

    # 全流程指标：ours不比B2差太多（容差0.5pp，含插值误差）
    rep = run(voyage, 60.0, "tables.npz")
    assert rep["ours"]["save_vs_B0"] >= rep["B2"]["save_vs_B0"] - 0.005, \
        f"排序异常 ours={rep['ours']['save_vs_B0']:.3f} B2={rep['B2']['save_vs_B0']:.3f}"
    # 锚点（转正口径：B2 2.95%为n=2可行真实成本，单台3.70%已作废）
    assert abs(rep["B0"]["fuel_t"] - 8.3667) < 0.05, f"B0锚点漂移 fuel={rep['B0']['fuel_t']}"
    assert abs(rep["B1"]["save_vs_B0"] * 100 - 6.02) < 0.03, f"B1锚点漂移 {rep['B1']['save_vs_B0']*100}"
    assert abs(rep["B2"]["save_vs_B0"] * 100 - 2.95) < 0.03, f"B2锚点漂移 {rep['B2']['save_vs_B0']*100}"
    assert abs(rep["ours"]["save_vs_B0"] * 100 - 6.93) < 0.03, f"ours锚点漂移 {rep['ours']['save_vs_B0']*100}"
    # rep键断言：B2多机核心指标；N=3全策略零过载
    assert abs(rep["B2"]["extra_g"] - 1932.1) < 5.0, f"B2 extra漂移 {rep['B2']['extra_g']}"
    assert rep["B2"]["n_switches"] >= 1, "B2开关数异常"
    for k in ("B0", "B1", "B2", "ours"):
        assert rep[k]["n_overload"] == 0, f"{k}过载非0"
    # dg投切阈值（1500→1 / 1501→2 / 4500→3 / 4501过载）
    assert DG.steady_n(1500) == 1, "steady1500"
    assert DG.steady_n(1501) == 2, "steady1501"
    assert DG.steady_n(4500) == 3, "steady4500"
    assert DG._candidates(4501) == [], "4501应无可行台数"
    assert DG.dg_opt(4501, 1, 60.0) == (3, 0.0), "4501过载分支"
    assert DG.dg_opt(1500, 1, 60.0)[0] == 1, "opt1500"
    assert DG.dg_opt(1501, 1, 60.0)[0] == 2, "opt1501"
    # eval插值回归（F1修正后：|dn|mean3.44/max36.0、|df|/f mean0.0001/max0.0009；阈值留10倍余量）
    en_mean, en_max, ef_mean, ef_max = interp_error()
    assert ef_mean < 0.001, f"插值回归超阈 ef_mean={ef_mean}"
    assert en_mean < 10.0, f"插值回归超阈 en_mean={en_mean}"
    assert rep["lookup_ms"] < 5.0, "查表耗时异常"
    # 非法输入守卫：期望抛ValueError
    for fn in (lambda: ctl.step(float("nan"), 60.0, 12.0),
               lambda: M._cl(float("nan"), 2.0, 25.0),
               lambda: M.friction_power(120.0, 0.0, 25.0),
               lambda: true_to_apparent(float("nan"), 0.0, 0.0, 5.0)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("NaN/非法输入未抛ValueError")
    print("ALL TESTS PASSED")
    for k in ("B0", "B1", "B2", "ours"):
        print(f"  {k}: save={rep[k]['save_vs_B0'] * 100:.2f}%")
    print(f"  B0 fuel={rep['B0']['fuel_t']:.4f}t B2 extra={rep['B2']['extra_g']:.1f}g sw={rep['B2']['n_switches']}")
    print(f"  interp |dn|mean={en_mean:.2f} max={en_max:.2f} |df|/f mean={ef_mean:.4f} max={ef_max:.4f}")


if __name__ == "__main__":
    main()
