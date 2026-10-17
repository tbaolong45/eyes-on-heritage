"""Shared paths, constants and helpers for the Eyes on Heritage pipeline."""
import os, json, random, hashlib, datetime
from pathlib import Path
import numpy as np

# ---------------- paths: edit here or set the environment variables ----------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace"))                               # folder holding the inputs below
IMAGERY_DIR = Path(os.environ.get("IMAGERY_DIR", DATA_DIR / "train_dataset"))           # <site-country>/YYYY_YYYYMMDD.tif
POINTS_FILE = Path(os.environ.get("POINTS_FILE", DATA_DIR / "train_points_organized.geojson"))
PAIRS_FILE = Path(os.environ.get("PAIRS_FILE") or next(
    (p for p in (DATA_DIR / "train_pairs.csv", DATA_DIR / "train_pairs.json") if p.exists()), DATA_DIR / "train_pairs.csv"))
EVAL_FILE = Path(os.environ.get("EVAL_FILE", DATA_DIR / "eval_original_776.csv"))       # optional: organizer labels for scoring
ROOT = Path(os.environ.get("HERITAGE_ROOT", DATA_DIR / "heritage_run"))                 # caches, reports, logs, models
RAW = ROOT / "data" / "raw"                                                             # p0_data.py links <site> -> IMAGERY_DIR folders
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for p in (RAW, CACHE, REPORTS, MODELS, CACHE / "tiles"):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = ["construction", "destruction", "soil_sealing", "no_change"]
C2I = {c: i for i, c in enumerate(CLASSES)}
NC = C2I["no_change"]
SITES = ["agadez", "aleppo", "derna", "marrakesh", "timbuktu"]

PX = 0.5            # metres / pixel
CELL_PX = 200       # 100 m cell
TILE = 384          # 192 m tile


TILE_DIR = Path(os.environ["HERITAGE_TILE_DIR"]) if os.environ.get("HERITAGE_TILE_DIR") else CACHE / "tiles"


def tile_path(site, pid):
    return TILE_DIR / site / f"{pid}.npz"


def set_tile_dir(p):
    global TILE_DIR
    TILE_DIR = Path(p)


def seed_all(s=0):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def parse_date(fname):
    """'2023_20230914.tif' -> (datetime.date, fractional year)."""
    s = Path(fname).stem.split("_")[1]
    d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    fy = d.year + (d.timetuple().tm_yday - 1) / 365.25
    return d, fy


def site_dates(site_dir):
    fs = sorted(Path(site_dir).glob("*.tif"))
    return fs, [parse_date(f.name) for f in fs]


LABEL_COLS = ["class", "y", "year", "year_before", "year_after", "train_weight"]


def load_points():
    """Points table. If HERITAGE_SHUFFLE=<seed> is set, all label/metadata columns are permuted jointly across
    points (label-shuffle sanity test); geometry, site, cells and imagery stay in place."""
    import pandas as pd
    df = pd.read_parquet(CACHE / "points.parquet")
    sh = os.environ.get("HERITAGE_SHUFFLE")
    if sh:
        perm = np.random.RandomState(int(sh)).permutation(len(df))
        df[LABEL_COLS] = df[LABEL_COLS].iloc[perm].to_numpy()
        df["y"] = df["y"].astype(int)
        for c in ("year", "year_before", "year_after", "train_weight"):
            df[c] = pd.to_numeric(df[c])
    return df


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=range(4), average="macro", zero_division=0)


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)
