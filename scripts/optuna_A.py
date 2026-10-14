"""Small Optuna search (<=30 trials) for Model A LightGBM; objective = S_proxy (LOSO-nested), seed 0.
Best params are re-run with 3 seeds and logged as A_lgb_opt. usage: optuna_A.py [point]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelA import PairTable, ModelA
from modelApt import PointTable, ModelApt
from cv import run_oof, evaluate, run_experiment
from pairs import make_folds
import pandas as pd, optuna

point = len(sys.argv) > 1 and sys.argv[1] == "point"
seed_all(0)
pts = load_points(); folds = make_folds(pts, 0)
U = pd.read_parquet(CACHE / "featA_tiles_unary.parquet"); P = pd.read_parquet(CACHE / "featA_tiles_pairs.parquet")
ex = pd.read_parquet(CACHE / "dinopair_dinov2-l_B1.parquet")
T = PairTable(pts, U, P, ex)
PT = PointTable(T) if point else None
base = dict(algo="lgb", agg="top2", k=2, reverse=False, ambiguous="exclude", label_w=True, balance=True,
            variant="tiles", dino="dinov2-l_B1", threads=4, point=point)
mk = (lambda c: (lambda: ModelApt(PT, c))) if point else (lambda c: (lambda: ModelA(T, c)))


def objective(trial):
    c = dict(base, lr=trial.suggest_float("lr", 0.01, 0.1, log=True),
             leaves=trial.suggest_int("leaves", 4, 31), min_leaf=trial.suggest_int("min_leaf", 10, 120),
             ff=trial.suggest_float("ff", 0.15, 0.8), l2=trial.suggest_float("l2", 0.1, 50, log=True),
             l1=trial.suggest_float("l1", 0.01, 5, log=True), depth=trial.suggest_int("depth", 3, 8))
    oi, _ = run_oof(mk(c), pts, "in", 0, folds, verbose=False)
    ol, _ = run_oof(mk(c), pts, "loso", 0, folds, verbose=False)
    r = evaluate(oi, ol, pts)
    trial.set_user_attr("M_in", r["M_in"]); trial.set_user_attr("M_out", r["M_out"])
    return r["S"]


st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial(dict(lr=0.03, leaves=(7 if point else 15), min_leaf=(15 if point else 40), ff=(0.3 if point else 0.5),
                      l2=(10.0 if point else 5.0), l1=(1.0 if point else 0.5), depth=(4 if point else 6)))
st.optimize(objective, n_trials=30)
print("best", st.best_value, st.best_params, st.best_trial.user_attrs, flush=True)
tag = "Apt_opt" if point else "A_lgb_opt"
json.dump(dict(best=st.best_value, params=st.best_params), open(REPORTS / f"optuna_{tag}.json", "w"), indent=1)
cfg = dict(base, **st.best_params, notes="optuna best (30 trials, seed 0)")
run_experiment(tag, cfg, mk(cfg), pts, seeds=[0, 1, 2], notes=cfg["notes"])
