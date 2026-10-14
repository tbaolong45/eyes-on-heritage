"""Co-registration v2: +-32 px search with NCC verification; tiles written to cache/tiles_v2 (same format)."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from imgproc import build_point_tiles
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
OUT = CACHE / "tiles_v2"
def work(args):
    site, rows = args
    fs, dates = site_dates(RAW / site); srcs = [rasterio.open(f) for f in fs]; stats = []
    for pid, col, row in rows:
        fn = OUT / site / f"{pid}.npz"
        r = build_point_tiles(srcs, col, row, max_shift=32, min_resp=0.08, verify=True, pad=40)
        np.savez_compressed(fn, tiles=r["tiles"], raw=r["raw"], bad=r["bad"], shifts=r["shifts"], ref=r["ref"],
                            quality=np.array(r["quality"], dtype=object), dates=np.array([d.isoformat() for d, _ in dates]),
                            fyear=np.array([fy for _, fy in dates]))
        nr = [k for k in range(len(fs)) if k != r["ref"]]
        stats += [dict(site=site, shift=float(np.hypot(*r["shifts"][k, :2])), applied=bool(r["shifts"][k, 2] > 0)) for k in nr]
    return stats
pts = load_points()
jobs = []
for s in SITES:
    (OUT / s).mkdir(parents=True, exist_ok=True)
    sp = pts[pts.site == s]
    rows = list(zip(sp.point_id, sp.col, sp.row))
    jobs += [(s, rows[i::4]) for i in range(4)]
with Pool(int(os.environ.get("FEAT_PROCS", 6))) as p:
    st = pd.DataFrame(sum(p.map(work, jobs), []))
print(st.groupby("site").agg(applied=("applied", "mean"), shift_med=("shift", "median"),
                             shift_p90_applied=("shift", lambda s: s[st.loc[s.index, "applied"]].quantile(0.9))).round(3))
