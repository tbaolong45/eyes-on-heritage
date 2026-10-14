"""Calibration + label-shift adaptation for the equal-weight ensemble, evaluated fully nested.
  TS   : per-member temperature scaling (NLL on held-in LOSO OOF), then equal-weight average
  BCTS : bias-corrected temperature scaling (Alexandari et al. 2020): softmax(log p / T + b_c)
  EM   : Saerens et al. (2002) prior re-estimation on the target site's *unlabelled* predictions
Every fitted quantity (T, b_c, decision t/b) uses held-in data only; EM uses only the target's predictions.
usage: calib_eval.py <member1,...> [--tag name]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
from common import *
from pairs import make_folds, tune_decision, decide
import numpy as np
from scipy.optimize import minimize


def fast_f1(y, p, per_class=False):
    cm = np.bincount(np.asarray(y) * 4 + np.asarray(p), minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    return f if per_class else f.mean()


common.macro_f1 = fast_f1
members = sys.argv[1].split(",")
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "calib"
pts = load_points(); y = pts.y.to_numpy(); site = pts.site.to_numpy(); fin = make_folds(pts, 0)["in"]
OL = np.stack([np.load(CACHE / "oof" / f"{m}.npz")["oof_loso"].mean(0) for m in members])
OI = np.stack([np.load(CACHE / "oof" / f"{m}.npz")["oof_in"].mean(0) for m in members])
LOG = lambda P: np.log(np.clip(P, 1e-6, 1))
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def fit_cal(P, yy, kind):
    """Return params for one member: TS -> (T,), BCTS -> (T, b0..b3)."""
    L = LOG(P)
    def nll(th):
        T = np.exp(th[0]); b = np.r_[th[1:], 0.0] if kind == "bcts" else 0
        Q = softmax(L / T + b)
        return -np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-9, 1)).mean()
    x0 = np.zeros(1 if kind == "ts" else 4)
    th = minimize(nll, x0, method="L-BFGS-B").x
    return th


def apply_cal(P, th, kind):
    T = np.exp(th[0]); b = np.r_[th[1:], 0.0] if kind == "bcts" else 0
    return softmax(LOG(P) / T + b)


ALPHA = 1.0     # MAP-EM: Dirichlet pseudo-counts = ALPHA * n * src_prior (shrinks the estimate toward the source prior)


def em_prior(P, src_prior, iters=100, tol=1e-6):
    """Saerens EM (MAP variant): adapt posteriors P (n,4) from source prior to the estimated target prior."""
    q = src_prior.copy(); n = len(P)
    for _ in range(iters):
        A = P * (q / src_prior)[None]; A /= A.sum(1, keepdims=True)
        qn = (A.sum(0) + ALPHA * n * src_prior) / (n + ALPHA * n)
        if np.abs(qn - q).max() < tol: q = qn; break
        q = qn
    A = P * (q / src_prior)[None]; return A / A.sum(1, keepdims=True), q


def ensemble(Pst, idx_fit, cal, apply_idx_sets):
    """Calibrate each member on idx_fit (LOSO OOF), then equal-weight average on each requested set."""
    outs = []
    for S_, ix in apply_idx_sets:
        acc = 0
        for k in range(len(members)):
            if cal == "none":
                acc = acc + S_[k][ix]
            else:
                th = fit_cal(OL[k][idx_fit], y[idx_fit], cal)
                acc = acc + apply_cal(S_[k][ix], th, cal)
        outs.append(acc / len(members))
    return outs


def run(cal, em):
    pred_out = np.zeros(len(y), int); pred_in = np.zeros(len(y), int); qs = {}
    for s in SITES:
        te = np.where(site == s)[0]; hi = np.where(site != s)[0]
        Phi, Pte = ensemble(OL, hi, cal, [(OL, hi), (OL, te)])
        if em:
            # source prior implied by the (calibrated) model on held-in data, not the label frequency:
            # the models were trained with class/site-balanced weights, so their posteriors encode a different prior
            src = Phi.mean(0)
            # held-in sites are adapted the same way (each site on its own predictions) before tuning t,b
            Phi_ad = Phi.copy()
            for r in SITES:
                if r == s: continue
                m = site[hi] == r; Phi_ad[m], _ = em_prior(Phi[m], src)
            Phi = Phi_ad; Pte, q = em_prior(Pte, src); qs[s] = q.round(3).tolist()
        t, b, _ = tune_decision(Phi, y[hi]); pred_out[te] = decide(Pte, t, b)
    for k in np.unique(fin):                    # M_in: no label-shift step (test cells come from training sites)
        te = np.where(fin == k)[0]; hi = np.where(fin != k)[0]
        Phi, Pte = ensemble(OL, hi, cal, [(OL, hi), (OI, te)])
        t, b, _ = tune_decision(Phi, y[hi]); pred_in[te] = decide(Pte, t, b)
    Mi, Mo = fast_f1(y, pred_in), fast_f1(y, pred_out)
    return dict(cal=cal, em=em, M_in=Mi, M_out=Mo, S=hm(Mi, Mo), perclass_out=fast_f1(y, pred_out, True).round(3).tolist(),
                per_site_out={s: round(float(fast_f1(y[site == s], pred_out[site == s])), 3) for s in SITES},
                est_prior_target=qs, true_prior_target={s: (np.bincount(y[site == s], minlength=4) / (site == s).sum()).round(3).tolist() for s in SITES})


def ece(P, yy, nb=15):
    c = P.max(1); ok = (P.argmax(1) == yy).astype(float); e = 0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(e)


rows = []
if "--alpha" in sys.argv: ALPHA = float(sys.argv[sys.argv.index("--alpha") + 1])
for cal in ("none", "ts", "bcts"):
    for em in (False, True):
        r = run(cal, em); rows.append(r)
        print(f"cal={cal:5s} em={em!s:5s}  M_in {r['M_in']:.4f}  M_out {r['M_out']:.4f}  S {r['S']:.4f}  aleppo {r['per_site_out']['aleppo']:.3f}", flush=True)
# calibration quality (nested ECE / NLL on LOSO)
calq = {}
for cal in ("none", "ts", "bcts"):
    P = np.zeros((len(y), 4))
    for s in SITES:
        te = np.where(site == s)[0]; hi = np.where(site != s)[0]
        P[te] = ensemble(OL, hi, cal, [(OL, te)])[0]
    calq[cal] = dict(ece=ece(P, y), nll=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()))
    print(f"calibration {cal}: LOSO ECE {calq[cal]['ece']:.4f} NLL {calq[cal]['nll']:.4f}")
json.dump(dict(members=members, alpha=ALPHA, rows=rows, calibration=calq), open(REPORTS / f"calib_{tag}.json", "w"), indent=1, default=float)
