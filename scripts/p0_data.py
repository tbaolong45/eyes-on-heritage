"""P0: data audit and tables, from the inputs configured in src/common.py.
  IMAGERY_DIR/<site-country>/YYYY_YYYYMMDD.tif  -> linked as RAW/<site>, audited (4-band uint8, 0.5 m, one grid per site),
                                                   100 m cell lattice from the withheld (alpha = 0) cells -> cache/site_meta.json
  POINTS_FILE (train_points_organized.geojson)  -> cache/points.parquet  (use_for_training points: class, year_before/after,
                                                   usable_years, reliability weight, cells and 200 m blocks for the CV folds)
  PAIRS_FILE  (train_pairs.csv / .json)         -> cache/pair_labels.parquet (every labelled year pair of those points)
  EVAL_FILE   (eval_original_776.csv, optional) -> y_orig = organizer label of the organizer points (scored by audit_ensemble.py)"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy import ndimage

for f in (POINTS_FILE, PAIRS_FILE):
    assert f.exists(), f"missing input {f} (set DATA_DIR or the path variables in src/common.py)"

# ---------- imagery: one folder of GeoTIFFs per site, linked as RAW/<site> ----------
site_dirs = {}
for d in sorted(Path(r) for r, _, fs in os.walk(IMAGERY_DIR, followlinks=True) if any(f.lower().endswith(".tif") for f in fs)):   # follows symlinked/mounted folders
    name = d.name.lower().replace("train_imagery_", "")
    site = next((s for s in SITES if name.startswith(s)), None)
    if site:
        assert site not in site_dirs, f"two imagery folders for {site}: {site_dirs.get(site)} and {d}"
        site_dirs[site] = d
missing = [s for s in SITES if s not in site_dirs]
assert not missing, f"no GeoTIFF folder found under {IMAGERY_DIR} for {missing} (expected e.g. {IMAGERY_DIR}/agadez-niger/2019_20191208.tif)"
for s, d in site_dirs.items():
    link = RAW / s
    if link.is_symlink() or link.exists():
        if link.resolve() == d.resolve(): continue
        link.unlink()
    link.symlink_to(d.resolve(), target_is_directory=True)
print("imagery:", {s: str(d) for s, d in site_dirs.items()})

audit, site_meta = [], {}
for site in SITES:
    fs, dates = site_dates(RAW / site)
    assert len(fs) >= 2, (site, "needs at least two dates")
    prof0 = None; alpha_or = None; valid_pct = []
    for f in fs:
        with rasterio.open(f) as r:
            assert r.count == 4 and r.dtypes[0] == "uint8", (f, r.count, r.dtypes)
            assert abs(r.res[0] - PX) < 1e-6 and abs(r.res[1] - PX) < 1e-6, (f, r.res)
            assert r.crs.is_projected, (f, r.crs)
            meta = (r.crs.to_string(), r.transform, r.width, r.height)
            prof0 = prof0 or meta
            assert meta == prof0, ("dates of one site are not on one grid", f)
            a = r.read(4) > 0
        valid_pct.append(a.mean() * 100); alpha_or = a if alpha_or is None else alpha_or | a
    crs, tr, W, H = prof0
    # withheld cells: alpha = 0 in every date and not connected to the border -> 100 m lattice offset
    lab, n = ndimage.label(~alpha_or)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    holes = [i for i in range(1, n + 1) if i not in border]; sl = ndimage.find_objects(lab)
    offs_c = [sl[i - 1][1].start % CELL_PX for i in holes] + [sl[i - 1][1].stop % CELL_PX for i in holes]
    offs_r = [sl[i - 1][0].start % CELL_PX for i in holes] + [sl[i - 1][0].stop % CELL_PX for i in holes]
    oc = int(np.bincount(offs_c, minlength=CELL_PX).argmax()) if holes else 0
    orr = int(np.bincount(offs_r, minlength=CELL_PX).argmax()) if holes else 0
    site_meta[site] = dict(crs=crs, transform=list(tr)[:6], width=W, height=H, lat_off_col=oc, lat_off_row=orr,
                           dates=[d.isoformat() for d, _ in dates], files=[f.name for f in fs])
    audit.append(dict(site=site, folder=str(site_dirs[site]), crs=crs, width=W, height=H, n_dates=len(fs),
                      dates=", ".join(d.isoformat() for d, _ in dates), valid_pct_min=round(min(valid_pct), 1),
                      n_withheld_cells=len(holes), lattice_offset=(oc, orr)))
    print(audit[-1], flush=True)
json.dump(site_meta, open(CACHE / "site_meta.json", "w"), indent=1)

# ---------- points ----------
gj = json.load(open(POINTS_FILE))
ev = pd.read_csv(EVAL_FILE) if EVAL_FILE.exists() else None
orig = dict(zip(ev.point_id, ev.label_original)) if ev is not None else {}
rows, skipped = [], 0
for f in gj["features"]:
    p = dict(f["properties"]); lon, lat = f["geometry"]["coordinates"][:2]
    if str(p.get("use_for_training", True)).lower() not in ("true", "1"):
        skipped += 1; continue
    site = p["site"].split("-")[0]; assert site in SITES, site
    src = int(p["point_id"])
    pid = f"{site}_{src:04d}" if p["source"] == "organizers" else f"{site}_x{src:05d}"
    m = site_meta[site]; x, y_ = Transformer.from_crs("OGC:CRS84", m["crs"], always_xy=True).transform(lon, lat)
    a, b, c, d, e, ff = m["transform"]; col = (x - c) / a; row = (y_ - ff) / e
    assert 0 <= col < m["width"] and 0 <= row < m["height"], ("point outside the site raster", pid)
    cr = int(np.floor((row - m["lat_off_row"]) / CELL_PX)); cc = int(np.floor((col - m["lat_off_col"]) / CELL_PX))
    usable = sorted(int(v) for v in str(p.get("usable_years") or "").split(",") if v.strip().isdigit())
    changed = p["class"] != "no_change"
    assert not changed or p.get("year_after") is not None, ("change point without year_after", pid)
    ya = float(p["year_after"]) if p.get("year_after") is not None else np.nan
    yb = float(p["year_before"]) if p.get("year_before") is not None else np.nan
    rows.append(dict(point_id=pid, src_index=src, site=site, lon=lon, lat=lat, col=col, row=row, utm_x=x, utm_y=y_,
                     cell_id=f"{site}_{cr}_{cc}", cell_r=cr, cell_c=cc, block_id=f"{site}_{cr // 2}_{cc // 2}",
                     source=p["source"], review=p.get("review_status"), **{"class": p["class"]}, y=C2I[p["class"]],
                     y_orig=(C2I[orig[src]] if p["source"] == "organizers" and src in orig else -1),
                     class_original=p.get("class_original"), year_before=yb, year_after=ya, year=(ya if changed else np.nan),
                     train_weight=float(p.get("weight", 1.0)), usable_years=",".join(map(str, usable)),
                     flag_definition=p.get("flag_definition")))
pts = pd.DataFrame(rows)
assert pts.point_id.is_unique
pts.to_parquet(CACHE / "points.parquet")
print(f"{len(pts)} training points ({skipped} not used) |", pts.source.value_counts().to_dict(), "|", pts["class"].value_counts().to_dict())
if ev is not None:
    print(f"organizer labels from {EVAL_FILE.name}: {(pts.y_orig >= 0).sum()} training points "
          f"({len(ev) - (pts.y_orig >= 0).sum()} evaluation points are not training points and are not scored)")

# ---------- labelled pairs ----------
pr = pd.read_json(PAIRS_FILE) if PAIRS_FILE.suffix == ".json" else pd.read_csv(PAIRS_FILE)
need = {"point_id", "site", "year1", "year2", "label", "weight"}
assert need <= set(pr.columns), f"{PAIRS_FILE} lacks columns {need - set(pr.columns)}"
pr["site_s"] = pr.site.str.split("-").str[0]
key = {(s, si): pid for pid, s, si in zip(pts.point_id, pts.site, pts.src_index)}
pr["pid"] = [key.get((s, int(p))) for s, p in zip(pr.site_s, pr.point_id)]
n_drop = int(pr.pid.isna().sum()); pr = pr[pr.pid.notna()].copy()
pw = dict(zip(pts.point_id, pts.train_weight))
lab = pd.DataFrame(dict(point_id=pr.pid, year1=pr.year1.astype(int), year2=pr.year2.astype(int), y=pr.label.map(C2I).astype(int),
                        # the CSV weight is point reliability x derivation; the point part is applied by point_weights()
                        w_pair=pr.weight.to_numpy(float) / pr.pid.map(pw).to_numpy(float),
                        adjacent=pr.get("adjacent", pd.Series(0, index=pr.index)).astype(int), derivation=pr.get("derivation", "")))
yrs = {s: {int(d[:4]) for d in m["dates"]} for s, m in site_meta.items()}
site_of = dict(zip(pts.point_id, pts.site))
bad = [(p, a, b) for p, a, b in zip(lab.point_id, lab.year1, lab.year2) if a not in yrs[site_of[p]] or b not in yrs[site_of[p]] or a >= b]
assert not bad, f"pairs with years missing from the imagery or not ordered: {bad[:5]}"
lab.to_parquet(CACHE / "pair_labels.parquet")

# whole scenes the review rejected for (nearly) every point of a site: predict.py drops them for test points of that site
for s, g in pts.groupby("site"):
    rej = [int(d[:4]) for d in site_meta[s]["dates"] if np.mean([d[:4] not in u.split(",") for u in g.usable_years]) >= 0.9]
    site_meta[s]["unusable_scenes"] = rej
json.dump(site_meta, open(CACHE / "site_meta.json", "w"), indent=1)
print("unusable scenes per site (rejected for >= 90 % of its points):", {s: m["unusable_scenes"] for s, m in site_meta.items()})
no_pairs = sorted(set(pts.point_id) - set(lab.point_id))
print(f"{len(lab)} labelled pairs ({n_drop} rows of non-training points dropped) |", lab.y.map(dict(enumerate(CLASSES))).value_counts().to_dict(),
      f"| training points without labelled pairs: {len(no_pairs)}")

with open(REPORTS / "data_audit.md", "w") as fh:
    fh.write(f"# Data audit\n\nInputs: `{IMAGERY_DIR}`, `{POINTS_FILE.name}`, `{PAIRS_FILE.name}`" + (f", `{EVAL_FILE.name}`" if ev is not None else "") + "\n\n")
    fh.write(pd.DataFrame(audit).to_markdown(index=False))
    fh.write(f"\n\nTraining points: {len(pts)} | cells {pts.cell_id.nunique()} | 200 m blocks {pts.block_id.nunique()} | labelled pairs {len(lab)}\n\n")
    fh.write(pd.crosstab(pts.site, pts["class"]).to_markdown() + "\n")
print(pd.crosstab(pts.site, pts["class"]).reindex(columns=CLASSES, fill_value=0))
