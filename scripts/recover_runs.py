"""Re-create runs.jsonl / results.md rows from saved OOF files (rows lost when an rsync overwrote reports/).
usage: recover_runs.py name1,name2,...   (set HERITAGE_SHUFFLE as for the original runs)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from cv import evaluate, log_result
import numpy as np
pts = load_points()
for n in sys.argv[1].split(","):
    z = np.load(CACHE / "oof" / f"{n}.npz")
    res = [evaluate(z["oof_in"][i], z["oof_loso"][i], pts) for i in range(len(z["oof_in"]))]
    log_result(n, {"recovered_from_oof": True}, list(range(len(res))), res,
               notes="row recovered from OOF file (original row lost to rsync overwrite); fit_info unavailable")
