# Eyes on Heritage: multi-temporal change classification

Point-level change detection on very-high-resolution (0.5 m) multi-temporal RGB satellite imagery. For every point the
pipeline outputs `p_change` and one class from `construction`, `destruction`, `soil_sealing`, `no_change`.

**Validation score (fully nested):** S = HM(M_in, M_out) = **0.642** (cell-bootstrap 95 % CI 0.599–0.680),
with M_in (spatial 5-fold CV, 200 m blocks) = 0.720 and M_out (leave-one-site-out) = 0.579.

| class | F1 in-domain | F1 leave-one-site-out |
|---|---|---|
| construction | 0.756 | 0.672 |
| destruction | 0.578 | 0.458 |
| soil_sealing | 0.667 | 0.400 |
| no_change | 0.880 | 0.787 |

This repository holds the code only. Data, caches, trained weights and reports are produced by the scripts below.

## Method
```
imagery stack + points
  └─ preprocessing: per point × date quality flags (sharpness, contrast, haze, valid alpha)
                    local co-registration (phase correlation on gradients, ≤ 12 px), 192 m tiles
       ├─ engineered pair features (Lab, change vector, SSIM, LBP, ExG, shadows, bare/paved) + DINOv2 distances
       │     ├─ Model A    LightGBM per pair of dates, top-2 pair aggregation (year-free)
       │     └─ Model A-pt LightGBM per point on statistics over all pairs
       ├─ frozen DINOv2-L tokens (8 dihedral views, canonical frame)
       │     └─ Model B    bitemporal transformer head, Gaussian centre pooling
       └─ Model C          siamese DINOv2-L + LoRA (r=16, last 8 blocks), heavy image augmentation
  equal-weight average of A, A-pt, B, C (seed-averaged)
  decision: p_change ≥ t → argmax change class + log-bias b, else no_change
  monotone Platt calibration of p_change for the reported probabilities
```
Annotation metadata (year, teams, ...) is never used as model input, only for sample weights. Pixels with alpha = 0
(outside the area or withheld test cells) are masked everywhere and never reconstructed.

## Repository layout
```
src/          library code: preprocessing, features, models A/A-pt/B/C (+ D, E experiments), validation, final training
scripts/      pipeline steps and experiments (see "Reproduce" below)
predict.py    end-to-end inference on new points / a new site
configs/      final member and ensemble configurations (equal weights, t, b, p_change calibration)
notebooks/    build_notebook.py writes a walk-through notebook of the submitted pipeline
```

## Installation
Python 3.12 and a CUDA GPU (developed on an RTX 5000 Ada, 32 GB).
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Backbone weights (`facebook/dinov2-large`) are downloaded from the Hugging Face Hub on first use. `HF_HOME` sets the
cache directory.

## Data
```
$DATA_DIR/train_annotations.geojson
$DATA_DIR/train_imagery/train_imagery_<site>-<country>.zip     # one GeoTIFF per date: YYYY_YYYYMMDD.tif
```
`DATA_DIR` defaults to `~/data/train_dataset`. `HERITAGE_ROOT` (default `~/heritage`) is where `cache/`, `reports/`,
`logs/` and `models/` are written. Clone the repository to that path, or point the variable at the clone.

## Reproduce training
`scripts/run.sh <log name> <script> [args]` runs one step and writes `logs/<log name>.log`.
```bash
R=scripts/run.sh
bash $R p0    scripts/p0_audit.py                   # unzip, asserts, points/cells/blocks, reports/data_audit.md
bash $R p1    scripts/p1_tiles.py                   # quality flags, co-registration, tile cache
bash $R featA scripts/p3_featA.py                   # Model A features
bash $R emb   scripts/p4_embed.py dinov2-l B1       # DINOv2-L token cache (~50 GB; put cache/emb on fast local disk)
bash $R A     scripts/exp_A.py A_lgb_top2_dino '{"agg":"top2","dino":"dinov2-l_B1"}'
bash $R Apt   scripts/optuna_A.py point             # A-pt with a 30-trial Optuna search -> run "Apt_opt"
bash $R B     scripts/exp_B.py B_l_B1_norev '{"agg":"top2","reverse":false}'
bash $R C     scripts/exp_C.py C_l_B2_lora3 '{}' 0,1,2
bash $R ens   scripts/ensemble.py A_lgb_top2_dino,Apt_opt,C_l_B2_lora3,B_l_B1_norev --weights eq --out ens_eq
bash $R audit scripts/audit_ensemble.py A_lgb_top2_dino,Apt_opt,C_l_B2_lora3,B_l_B1_norev --tag ens5
bash $R final scripts/train_final.py reports/ens_eq.json      # all 776 points, 5 seeds per member -> models/final/
cp configs/ensemble.json models/final/ensemble.json            # equal weights, t, b and p_change calibration
```
Each experiment appends a row to `reports/results.md` and saves out-of-fold predictions to `cache/oof/`. Reports and
figures: `scripts/report_figs.py ens_eq`, `scripts/final_report.py ens_eq --audit audit_ens5.json`, `scripts/eda.py`.
`python notebooks/build_notebook.py` writes the walk-through notebook. With `MODE = "load"` it recomputes the scores from
the cached out-of-fold predictions.

## Predict
```bash
python predict.py --points test_points.geojson --imagery /path/to/site_imagery --out submission.csv
# options: --site NAME  --id-col id  --use-year (score only the core pair when a 'year' column is given)
#          --models DIR (default $HERITAGE_ROOT/models/final)  --work DIR
```
The points file is GeoJSON or CSV with lon/lat (CRS84) and an id column. The imagery is a folder or zip of
`YYYY_YYYYMMDD.tif` files on one grid. The UTM zone is read from the GeoTIFF CRS, and any number of dates works. Output columns:
`id, p_change, pred_class, p_construction, p_destruction, p_soil_sealing, p_no_change`. They are defined in the `OUTPUT`
block at the top of `predict.py`, so the official submission format can be mapped there.

## Validation protocol
* In-domain proxy: `StratifiedGroupKFold(5)` with groups = 200 m blocks, stratified by site × class.
* Out-of-domain proxy: leave-one-site-out, the closest analogue to an unseen city.
* Early stopping and checkpoint selection use an inner split of the training sites only. Every split is asserted disjoint.
* Ensemble weights are equal (not fitted). Threshold and class biases are fitted only on held-in data, which makes the score nested.
* Sanity checks: a label-shuffle test (`scripts/shuffle_test.sh`, `scripts/shuffle_eval.sh`) gives S ≈ 0.245, the chance
  level. Predicting a held-out site with `predict.py` reproduces its out-of-fold score.

## External resources
* DINOv2 ViT-L/14 (`facebook/dinov2-large`, Apache-2.0). Experiments also used DINOv2-g, DINOv3 (`timm` releases, DINOv3
  licence) and SAM ViT-H (Apache-2.0); these are not part of the submitted ensemble.
* PyTorch, transformers, timm, peft, LightGBM, CatBoost, scikit-learn, scikit-image, OpenCV, rasterio, pyproj, shapely,
  geopandas, pandas, optuna, matplotlib.
