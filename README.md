# Eyes on Heritage: multi-temporal change classification

Point-level change detection on very-high-resolution (0.5 m) multi-temporal RGB satellite imagery. For every point the
pipeline outputs `p_change` and one class from `construction`, `destruction`, `soil_sealing`, `no_change`.

## Results (fully nested validation)
S = harmonic mean of M_in (spatial 5-fold CV, 200 m blocks) and M_out (leave-one-site-out), macro-F1 over 4 classes.

| | v1 | **v3 (current)** |
|---|---|---|
| labels | `train_annotations.geojson`, 776 points | `training_points_reviewed.geojson`, 932 points |
| members | A, A-pt, B, C (DINOv2-L) | A, B, C (DINOv2-L), **C (DINOv3-sat-L)** |
| M_in | 0.720 | 0.692 |
| M_out | 0.579 | **0.590** |
| S | 0.642 (CI 0.599–0.680) | 0.637 (CI 0.598–0.671) |
| Aleppo (closest proxy to an unseen dense city) | 0.652 | **0.684** |

Comparison at the same number of CV seeds for Model C (one seed): v1 0.618 vs v3 **0.637**. The final models of both
versions are trained with 5 seeds per member. The v3 score on the 776 organizer points with their original labels is 0.632.

Per-class F1 (v3, nested):

| class | in-domain | leave-one-site-out |
|---|---|---|
| construction | 0.748 | 0.694 |
| destruction | 0.577 | 0.440 |
| soil_sealing | 0.595 | 0.458 |
| no_change | 0.848 | 0.769 |

## Method (v3)
```
imagery stack + points
  └─ preprocessing: per point × date quality flags (sharpness, contrast, haze, valid alpha)
                    local co-registration (phase correlation on gradients, ≤ 12 px), 192 m tiles
       ├─ engineered pair features (Lab, change vector, SSIM, LBP, ExG, shadows, bare/paved) + DINOv2 distances
       │     └─ Model A   LightGBM per pair of dates, top-2 pair aggregation (year-free)
       ├─ frozen DINOv2-L tokens (8 dihedral views, canonical frame)
       │     └─ Model B   bitemporal transformer head, Gaussian centre pooling
       ├─ Model C   siamese DINOv2-L + LoRA (r=16, last 8 blocks), heavy image augmentation
       └─ Model C3  siamese DINOv3-sat-L (SAT-493M) + LoRA, same head and augmentation
  equal-weight average of A, B, C, C3 (5 seeds each)
  decision: p_change ≥ t (0.74) → argmax change class + log-bias b, else no_change
  monotone Platt calibration of p_change for the reported probabilities (LOSO ECE 0.235 → 0.038)
```
* **Labels v3**: pairs of dates are labelled from `year_before` / `year_after`. A changed point is positive for pairs that
  bracket the change, and pairs inside the bracket are ignored. Reviewer `train_weight` is the reliability weight, multiplied
  by (site × class) balancing.
* Annotation metadata is never used as model input, only for sample weights. Pixels with alpha = 0 (outside the area
  or withheld test cells) are masked everywhere and never reconstructed. No training point lies in a withheld cell
  (`scripts/check_holes_v2.py`).

### Classification heads
* **A**: LightGBM multiclass on pair features. Pair probabilities are averaged over the two most-changed pairs.
* **B / C / C3**: shared head design. Tokens are projected to 256-d, then fused per token as [h_b, h_a, |h_a − h_b|, cos],
  plus a learned 2-D position embedding, a 2-layer transformer (4 heads, pre-norm), and attention pooling with a learnable
  Gaussian prior around the point (σ initialised at 16 m). The output is a linear layer to 4 logits. Loss: focal (γ = 1.5),
  label smoothing 0.05.

## Repository layout
```
src/          preprocessing, features, models A/A-pt/B/C (+ D, E experiments), validation, final training / prediction
scripts/      pipeline steps and experiments
predict.py    end-to-end inference on new points / a new site
configs/      v3 member and ensemble configurations (configs/v1/: previous version)
notebooks/    build_notebook.py writes a walk-through notebook (v1 pipeline)
```

## Installation
Python 3.12 and a CUDA GPU (developed on an RTX 5000 Ada, 32 GB).
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Backbones are downloaded from the Hugging Face Hub on first use: `facebook/dinov2-large` and
`timm/vit_large_patch16_dinov3.sat493m`. `HF_HOME` sets the cache directory.

## Data
```
$DATA_DIR/train_annotations.geojson
$DATA_DIR/training_points_reviewed.geojson
$DATA_DIR/train_imagery/train_imagery_<site>-<country>.zip     # one GeoTIFF per date: YYYY_YYYYMMDD.tif
```
`DATA_DIR` defaults to `~/data/train_dataset`. `HERITAGE_ROOT` is where `cache/`, `reports/`, `logs/` and `models/`
are written. v1 uses `~/heritage`. v3 uses a second root (`~/heritage_v2`) that links `data/raw` and `cache/tiles` from
the v1 root.

## Reproduce v3
```bash
# v1 root first: unzip + audit, tiles for the organizer points
R=scripts/run.sh
bash $R p0 scripts/p0_audit.py
bash $R p1 scripts/p1_tiles.py
# v3 root
mkdir -p ~/heritage_v2/{cache,data,logs,reports,models}
ln -sfn ~/heritage/data/raw ~/heritage_v2/data/raw; ln -sfn ~/heritage/cache/tiles ~/heritage_v2/cache/tiles
cp ~/heritage/cache/site_meta.json ~/heritage_v2/cache/
V=scripts/run_v2.sh
bash $V points scripts/p0_points_v2.py              # reviewed labels -> points table (year_before/after, train_weight)
bash $V p1     scripts/p1_tiles.py                  # tiles for the extra points (existing tiles are reused)
bash $V holes  scripts/check_holes_v2.py            # no point inside a withheld cell
bash $V featA  scripts/p3_featA.py
bash $V emb    scripts/p4_embed.py dinov2-l B1      # token cache, ~60 GB: put cache/emb on a fast local disk
bash $V A      scripts/exp_A.py A_v3lab_dinov2 '{"agg":"top2","dino":"dinov2-l_B1"}'
bash $V B      scripts/exp_B.py B_v3lab_dinov2 '{"agg":"top2","reverse":false,"emb":"dinov2-l_B1"}'
bash $V C      scripts/exp_C.py C_v2lab_s0 '{}' 0
bash $V C3     scripts/exp_C.py C_v3lab_dinov3_s0 '{"bb":"dinov3s-l"}' 0
M=A_v3lab_dinov2,B_v3lab_dinov2,C_v2lab_s0,C_v3lab_dinov3_s0
bash $V ens    scripts/ensemble.py $M --weights eq --out ens_E3       # t, b on pooled LOSO OOF
bash $V calib  scripts/v2_ensemble.py $M --tag E3                     # p_change calibration (+ ensemble-level checks)
bash $V audit  scripts/audit_ensemble.py $M --tag E3                  # nested scores, leak assertions, bootstrap CI
bash $V final  scripts/train_final.py ~/heritage_v2/reports/ens_E3.json   # all points, 5 seeds per member
cp configs/ensemble.json ~/heritage_v2/models/final/ensemble.json        # weights, t, b, p_change calibration
```
v1 commands: `configs/v1/` holds its configuration. The steps match the ones above, run on the v1 root with
`scripts/run.sh`, and use members `A_lgb_top2_dino, Apt_opt, C_l_B2_lora3, B_l_B1_norev`.

## Predict
```bash
HERITAGE_ROOT=~/heritage_v2 python predict.py --points test_points.geojson --imagery /path/to/site_imagery --out submission.csv
# options: --site NAME  --id-col id  --use-year  --tta N (default 8; lower is faster)  --models DIR  --work DIR
```
The points file is GeoJSON or CSV with lon/lat (CRS84) and an id column. The imagery is a folder or zip of
`YYYY_YYYYMMDD.tif` files on one grid. The UTM zone is read from the GeoTIFF CRS, and any number of dates works. Output columns:
`id, p_change, pred_class, p_construction, p_destruction, p_soil_sealing, p_no_change`. They are defined in the `OUTPUT`
block at the top of `predict.py`. With two LoRA members and 8× test-time augmentation, about 140 points take
roughly 75 minutes on one GPU.

## Validation protocol
* In-domain proxy: `StratifiedGroupKFold(5)` with groups = 200 m blocks, stratified by site × class.
* Out-of-domain proxy: leave-one-site-out.
* Early stopping and checkpoint selection use an inner split of the training sites only. Every split is asserted disjoint.
* Ensemble weights are equal (not fitted). Threshold and class biases are fitted only on held-in data, which makes the score nested.
* Sanity checks: label-shuffle test (`scripts/shuffle_test.sh`, `scripts/shuffle_eval.sh`) gives S ≈ 0.245, the chance
  level. In-sample predictions on a training site are high but not perfect.

## Experiments that did not improve the nested ensemble
Frozen DINOv3-sat or RemoteCLIP features for A / A-pt / B (frozen DINOv2 is better). DINOv2-giant. SAM ViT-H
(AnyChange-style) features. Siamese U-Net with a ConvNeXt-DINOv3 encoder. Per-site feature / token normalisation.
Confident-learning label filtering. Wider co-registration search. EM label-shift adaptation. Stacking. Fitted ensemble
weights.

## External resources
* DINOv2 ViT-L/14 (`facebook/dinov2-large`, Apache-2.0). DINOv3 ViT-L/16 SAT-493M (`timm` release, DINOv3 licence).
* Also used in experiments: DINOv2-g, DINOv3 web / ConvNeXt (`timm`), SAM ViT-H (Apache-2.0), RemoteCLIP ViT-L/14.
* PyTorch, transformers, timm, peft, open_clip, LightGBM, CatBoost, scikit-learn, scikit-image, OpenCV, rasterio, pyproj,
  shapely, geopandas, pandas, optuna, matplotlib.
