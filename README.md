# Eyes on Heritage: multi-temporal change classification

Point-level change classification on 0.5 m multi-date RGB satellite imagery. For every point the pipeline outputs `p_change` and one
class from `construction`, `destruction`, `soil_sealing`, `no_change`. It is scored with the hackathon metric
**S = HM(M_in, M_out)**: macro-F1 under spatial cross-validation (M_in) and leave-one-site-out (M_out).

This version trains on the **organized dataset**: `train_points_organized.geojson` (1029 usable reviewed points) and
`train_pairs.csv` (19 045 labelled year pairs). `eval_original_776.csv` adds a score on the organizers' original labels.

## 1. Inputs
Put the data in one folder, `/workspace` by default:
```
/workspace/
├── train_points_organized.geojson      # reviewed points: class, year_before/after, usable_years, weight, use_for_training
├── train_pairs.csv                     # labelled year pairs per point (train_pairs.json with the same columns also works)
├── eval_original_776.csv               # optional: organizers' original labels (scored in the audit)
└── train_dataset/
    ├── agadez-niger/        2018_20180825.tif  2019_20191208.tif  ...
    ├── aleppo-syria/        ...
    ├── derna-libya/         ...
    ├── marrakesh-morocco/   ...
    └── timbuktu-mali/       ...        # one 4-band uint8 GeoTIFF per date, YYYY_YYYYMMDD.tif, all dates on one grid
```
Site folders are matched by their name prefix (`agadez…`, `aleppo…`, `derna…`, `marrakesh…`, `timbuktu…`), at any depth below
`train_dataset/`. Unzip the organizers' zips into it if needed: `unzip -q '*.zip' -d /workspace/train_dataset`.

## 2. Configure
All paths are set in one block at the top of `src/common.py`. You can edit that block or set environment variables:

| variable | default | what |
|---|---|---|
| `DATA_DIR` | `/workspace` | folder holding the inputs above |
| `IMAGERY_DIR` | `$DATA_DIR/train_dataset` | site folders of GeoTIFFs |
| `POINTS_FILE` | `$DATA_DIR/train_points_organized.geojson` | reviewed points |
| `PAIRS_FILE` | `$DATA_DIR/train_pairs.csv`, else `.json` | labelled pairs |
| `EVAL_FILE` | `$DATA_DIR/eval_original_776.csv` | organizer labels (optional) |
| `HERITAGE_ROOT` | `$DATA_DIR/heritage_run` | everything the pipeline writes: `cache/`, `reports/`, `logs/`, `models/` |

`HERITAGE_ROOT` needs about 100 GB free: the DINOv2 token cache alone is about 60 GB.

## 3. Install
Python 3.12 and a CUDA GPU (24 GB or more for Models C and C3).
```bash
git clone https://github.com/dangaura-tech/eyes-on-heritage.git /workspace/eyes-on-heritage
cd /workspace/eyes-on-heritage
bash scripts/setup_env.sh          # .venv with CUDA torch + requirements.txt
```
On a box that already has CUDA torch, you can skip the venv: `grep -v -E '^(torch|torchvision)==' requirements.txt | pip install -r /dev/stdin`.
Backbones download from the Hugging Face Hub on the first run: `facebook/dinov2-large` and `timm/vit_large_patch16_dinov3.sat493m`.
If a download is refused, accept the model's licence on its Hub page and run `huggingface-cli login`.

## 4. Run
```bash
cd /workspace/eyes-on-heritage
nohup bash scripts/run_all.sh > /workspace/run_all.out 2>&1 &
tail -f /workspace/run_all.out
```
`run_all.sh` runs every step in order and skips a step whose output already exists, so you can re-run it after a crash or a disconnect.
Each step logs to `$HERITAGE_ROOT/logs/<step>.log` and ends with `EXIT 0` on success. To run a single step:
`bash scripts/run.sh <log name> scripts/<script>.py [args]`.

| step | script | output | notes |
|---|---|---|---|
| p0 | `p0_data.py` | `cache/points.parquet`, `cache/pair_labels.parquet`, `cache/site_meta.json`, `reports/data_audit.md` | links the site folders, audits the imagery, builds the tables |
| p1 | `p1_tiles.py` | `cache/tiles/<site>/<point>.npz`, `cache/quality.parquet` | co-registered 192 m tiles; valid dates = `usable_years` |
| holes | `check_holes_v2.py` | – | asserts no training point lies in a withheld (alpha = 0) cell |
| featA | `p3_featA.py` | `cache/featA_tiles_*.parquet` | handcrafted pair features (CPU, all cores) |
| emb | `p4_embed.py dinov2-l B1` | `cache/emb/dinov2-l_B1/`, `cache/dinopair_*.parquet` | frozen DINOv2-L tokens, 8 views (GPU) |
| A | `exp_A.py` | `cache/oof/A_train_pairs_dinov2.npz` | LightGBM on pair features + DINOv2 distances |
| B | `exp_B.py` | `cache/oof/B_train_pairs_dinov2.npz` | transformer head on frozen tokens (GPU) |
| C | `exp_C.py` | `cache/oof/C_train_pairs_dinov2.npz` | siamese DINOv2-L + LoRA (GPU, hours per seed) |
| C3 | `exp_C.py {"bb":"dinov3s-l"}` | `cache/oof/C_train_pairs_dinov3.npz` | siamese DINOv3-SAT-L + LoRA (GPU, hours per seed) |
| ens | `ensemble.py --weights eq` | `reports/ens_<TAG>.json` | equal weights; threshold t and class biases b tuned on pooled leave-one-site-out OOF |
| calib | `v2_ensemble.py` | `reports/v2ens_<TAG>.json` | monotone Platt calibration of `p_change` |
| audit | `audit_ensemble.py` | `reports/audit_ens_<TAG>.md` | **the headline scores**: fully nested S with CI, per class, per site, and S on the organizers' original labels |
| final | `train_final.py` | `models/final/<member>/`, `models/final/ensemble.json` | all training points, 5 seeds per member |

Every experiment also appends a row to `reports/results.md`. Variables you can set before `run_all.sh`:
`MEMBERS` (default `A_train_pairs_dinov2,B_train_pairs_dinov2,C_train_pairs_dinov2,C_train_pairs_dinov3`), `SEEDS_AB` (`0,1,2`), `SEEDS_C` (`0`),
`FINAL_SEEDS` (`0,1,2,3,4`), `TAG` (defaults to `PAIRS_FROM`), `PAIRS_FROM` (`train_pairs`, see section 6). For a fast first pass, run only A and B:
`MEMBERS=A_train_pairs_dinov2,B_train_pairs_dinov2 TAG=quick bash scripts/run_all.sh`. Use a different `TAG` (or delete the `ens/calib/audit` reports) when you change `MEMBERS`, because finished steps are skipped.
Changing `PAIRS_FROM` needs nothing else: members and reports carry the label source in their names, and tiles, features and
embeddings are reused. `models/final/` always holds the last finished `final` step.

## 5. Predict
```bash
python predict.py --points test_points.geojson --imagery /path/to/site_folder_or_zip --out submission.csv
# options: --site NAME  --id-col id  --use-year  --tta N (default 8; lower is faster)  --models DIR  --work DIR
```
Set the same `DATA_DIR` / `HERITAGE_ROOT` as for training: the models are read from `$HERITAGE_ROOT/models/final`. The points file is
GeoJSON or CSV with lon/lat (CRS84) and an id column. The imagery is one site's folder or zip of `YYYY_YYYYMMDD.tif` files on one
grid. Test points have no reviewed `usable_years`, so their dates are filtered by the automatic quality flags. For a training site,
`predict.py` also drops the scenes the review rejected for the whole site (`unusable_scenes` in `cache/site_meta.json`: Agadez 2018
and 2024, Derna 2024, Marrakesh 2026), which the automatic flags often keep. Output columns are
`id, p_change, pred_class, p_construction, p_destruction, p_soil_sealing, p_no_change`; edit the `OUTPUT` block at the top of
`predict.py` to match the official submission format.

## 6. How the training data is used
* **Points**: `use_for_training = true` points only (1029). `weight` is the point's reliability weight, multiplied by
  (site × class) balancing. Organizer points keep the id `<site>_<point_id>`; team-added points are `<site>_x<point_id>`.
* **Pairs**: every model trains on the labelled pairs of `train_pairs` (`src/pairs.py: label_pairs`). A pair's weight is its
  derivation factor (event 1, before the event ×0.5, after it ×0.4). The point's reliability is applied once, through
  `point_weights`. Adjacent pairs that span a change count double as the "core" pair. Models with `reverse` also train on the
  time-reversed pairs (no_change stays no_change, construction ↔ destruction at half weight).
* **Dates**: a training point's valid dates are its reviewed `usable_years`, so the README's unusable scenes (Derna 2024,
  Marrakesh 2026, Agadez 2018/2024, local clouds) are never used. Prediction scores every pair of valid dates and takes the mean
  `p_change` of the two most changed pairs (year-free, the test condition).
* **Fallback**: a point without labelled pairs gets pairs from `year_before` / `year_after`, as in the previous version.
* **Label source switch** `PAIRS_FROM`: `train_pairs` (default, all labelled pairs), `train_pairs_event` (without the no_change
  pairs taken before / after a change point's event), or `years` (the previous `year_before` / `year_after` rule for every point;
  valid dates still follow `usable_years`). On a CPU check with Model A (no DINOv2 distances, 3 seeds) the three were within
  the hackathon's 0.02 noise margin: S 0.405 / 0.390 / 0.415, with equal M_out for `train_pairs` and `years` and a higher M_in
  for `years`. To decide on the full features, compare Model A once per source (minutes each, after the `emb` step):
  `for v in train_pairs years; do bash scripts/run.sh cmp_$v scripts/exp_A.py Acmp_$v "{\"agg\":\"top2\",\"dino\":\"dinov2-l_B1\",\"pairs_from\":\"$v\"}" 0,1,2; done`
  then read the two `Acmp_` rows of `reports/results.md`, and run the pipeline with the better one: `PAIRS_FROM=years bash scripts/run_all.sh`.

## 7. Validation protocol
* M_in: `StratifiedGroupKFold(5)` with groups = 200 m blocks, stratified by site × class. All pairs of a point stay in its fold.
* M_out: leave-one-site-out.
* Early stopping uses an inner split of the training part only. Every split is asserted disjoint (`audit_ensemble.py`).
* Ensemble weights are equal, not fitted. The threshold and class biases are fitted on held-in data only, so the audited score is nested.
* The audit also reports `S_org`: the same nested predictions scored against `label_original` from `eval_original_776.csv` on the
  728 organizer points that are training points. The 48 evaluation points with `use_for_training = false` are not scored.
* Sanity check: `scripts/shuffle_test.sh` retrains on permuted labels (the score must fall to chance, S ≈ 0.25). Its member names
  are those of the previous version; edit them before use.

## 8. Method
```
imagery stack + points
  └─ preprocessing: per point × date quality flags, local co-registration (phase correlation on gradients, ≤ 12 px), 192 m tiles
       ├─ engineered pair features (Lab, change vector, SSIM, LBP, ExG, shadows, bare/paved) + DINOv2 distances
       │     └─ Model A   LightGBM per pair of dates
       ├─ frozen DINOv2-L tokens (8 dihedral views)
       │     └─ Model B   bitemporal transformer head, Gaussian centre pooling
       ├─ Model C   siamese DINOv2-L + LoRA (r = 16, last 8 blocks), heavy image augmentation
       └─ Model C3  siamese DINOv3-SAT-L + LoRA, same head and augmentation
  pair probabilities -> point: mean p_change of the top-2 pairs; equal-weight average of the members
  decision: p_change ≥ t -> argmax change class + log-bias b, else no_change; monotone Platt calibration of p_change
```

## 9. What changed in this version
* `scripts/p0_data.py` replaces `p0_audit.py` and `p0_points_v2.py`. It needs neither `train_annotations.geojson` nor zipped
  imagery, and it builds one workspace instead of the earlier `~/heritage` + `~/heritage_v2` pair.
* Pair labels come from `train_pairs` and valid dates from `usable_years` (section 6); `PAIRS_FROM` switches the label source.
* `predict.py` drops a known site's reviewed unusable scenes for test points.
* `train_final.py --calib` writes the `p_change` calibration into `models/final/ensemble.json`. You no longer copy a fixed
  `configs/ensemble.json`.
* `scripts/run.sh` works from any clone location; `scripts/run_all.sh` is the one-command pipeline.
* `configs/reference_v3_932pts/` keeps the configurations and scores of the previous run (932 points, S = 0.637) for comparison.
  They are not read by the pipeline.

## Repository layout
```
src/          preprocessing, features, models A/A-pt/B/C (+ D, E experiments), validation, final training / prediction
scripts/      pipeline steps (run_all.sh, run.sh, p0_data.py, p1_tiles.py, ...) and experiments
predict.py    end-to-end inference on new points / a new site
configs/      reference configurations of the previous run
notebooks/    build_notebook.py writes the walk-through notebook of the v1 pipeline (it expects the v1 inputs)
```

## Troubleshooting
* **A step failed**: read `$HERITAGE_ROOT/logs/<step>.log`, fix the cause, run `run_all.sh` again; finished steps are skipped.
* **Out of GPU memory in C / C3**: add `\"bs\":16` to `C_CFG` and `C3_CFG` in `run_all.sh` (default batch 32).
* **DataLoader workers crash or the box has few CPUs**: add `\"workers\":2` to the member's config in `run_all.sh`.
* **macOS only**: LightGBM and scikit-learn bundle different OpenMP runtimes and can segfault together; run with
  `OMP_NUM_THREADS=1` and `"threads":1` in `A_CFG`. Linux GPU boxes are not affected.
* **Disk full during `emb`**: point `HERITAGE_ROOT` at a larger disk, or symlink `$HERITAGE_ROOT/cache/emb` to one.
* **`no GeoTIFF folder found`**: check that the five site folders sit under `IMAGERY_DIR` and contain the `.tif` files directly.

## External resources
DINOv2 ViT-L/14 (`facebook/dinov2-large`, Apache-2.0). DINOv3 ViT-L/16 SAT-493M (`timm` release, DINOv3 licence). PyTorch,
transformers, timm, peft, LightGBM, CatBoost, scikit-learn, scikit-image, OpenCV, rasterio, pyproj, pandas.
