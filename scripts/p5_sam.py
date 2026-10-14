"""SAM (ViT-H) AnyChange-style pair features for all training points -> cache/sampair.parquet"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from samfeat import sam_pair_table
import numpy as np
pts = load_points()
if "--smoke" in sys.argv: pts = pts.iloc[:4]
df = sam_pair_table(pts, lambda s, p: np.load(tile_path(s, p), allow_pickle=True)["tiles"], log=lambda m: print(m, flush=True))
print(df.describe().T.round(3).to_string())
if "--smoke" not in sys.argv:
    df.to_parquet(CACHE / "sampair.parquet")
