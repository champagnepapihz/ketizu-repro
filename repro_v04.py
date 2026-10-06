"""v0.4 PDF数字复现入口。工作目录=本文件夹。
依赖：numpy/scipy/netCDF4。数据：../data/era5_ras_zhoushan_2024.nc（不进包）。
"""
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from scheduler.synth import make_voyage
from scheduler.simulate import run
from scheduler.voyage_era5 import make_voyage_era5


def main():
    print("== 合成基准 ==")
    voyage, dt = make_voyage()
    rep = run(voyage, dt, os.path.join(BASE, "tables.npz"))
    for k, lbl in (("B0", "B0"), ("B1", "B1"), ("B2", "B2"), ("ours", "B3")):
        print(f"{lbl}: fuel={rep[k]['fuel_t']:.4f}t save={rep[k]['save_vs_B0']*100:.4f}%")
    print(f"pp(B3-B1)={(rep['ours']['save_vs_B0']-rep['B1']['save_vs_B0'])*100:.4f}")

    print("== ERA5真实风2024 ==")
    # nc 用相对路径：netCDF4 的 C 层打不开含中文的绝对路径，相对路径为纯 ASCII
    nc = os.path.join("..", "data", "era5_ras_zhoushan_2024.nc")
    voy, dt5 = make_voyage_era5(nc, loop=True)
    print(f"N={len(voy)} dt={dt5}")
    rep5 = run(voy, dt5, os.path.join(BASE, "tables.npz"))
    for k, lbl in (("B0", "B0"), ("B1", "B1"), ("B2", "B2"), ("ours", "B3")):
        print(f"{lbl}: fuel={rep5[k]['fuel_t']:.4f}t save={rep5[k]['save_vs_B0']*100:.4f}%")
    print(f"pp(B3-B1)={(rep5['ours']['save_vs_B0']-rep5['B1']['save_vs_B0'])*100:.4f}%")


if __name__ == "__main__":
    main()
