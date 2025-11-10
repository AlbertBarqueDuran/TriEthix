import argparse
import json
import os
import re
from math import atanh, tanh

import numpy as np
import pandas as pd


def base_label(label: str) -> str:
    return re.sub(r'_(2|3)$', '', label)


def fisher_mean_r(r_values):
    zs = []
    for r in r_values:
        if r is None or np.isnan(r):
            continue
        if abs(r) >= 1:
            r = np.nextafter(1.0, 0.0) if r > 0 else np.nextafter(-1.0, 0.0)
        zs.append(atanh(r))
    if not zs:
        return float("nan")
    return tanh(sum(zs) / len(zs))


def icc_3(data):
    X = np.asarray(data, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")

    mean_per_subject = X.mean(axis=1, keepdims=True)
    mean_per_rater = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()

    ss_total = ((X - grand_mean) ** 2).sum()
    ss_rows = (k * ((mean_per_subject - grand_mean) ** 2)).sum()
    ss_cols = (n * ((mean_per_rater - grand_mean) ** 2)).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc31 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    icc3k = (ms_rows - ms_error) / ms_rows
    return float(icc31), float(icc3k)


def safe_r(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_complete_triplets(estimates_path):
    with open(estimates_path, "r") as f:
        data = json.load(f)
    raw = data["models_raw"]

    buckets = {}
    for obj in raw:
        lbl = obj["label"]
        base = base_label(lbl)
        idx = 2 if lbl.endswith("_2") else 3 if lbl.endswith("_3") else 1
        buckets.setdefault(base, {})[idx] = obj

    complete = {b: runs for b, runs in buckets.items() if set(runs.keys()) == {1, 2, 3}}

    bases = sorted(complete.keys())
    return bases, complete


def compute_overall(bases, complete):
    metrics = ["virtue", "deontology", "consequentialism", "flip_rate"]

    series = {m: {1: [], 2: [], 3: []} for m in metrics}
    for b in bases:
        runs = complete[b]
        for r in (1, 2, 3):
            series["virtue"][r].append(runs[r]["weights"]["virtue"])
            series["deontology"][r].append(runs[r]["weights"]["deontology"])
            series["consequentialism"][r].append(runs[r]["weights"]["consequentialism"])
            series["flip_rate"][r].append(runs[r]["flip_rate"])

    rows = []
    for m in metrics:
        arr1 = np.array(series[m][1], float)
        arr2 = np.array(series[m][2], float)
        arr3 = np.array(series[m][3], float)

        r12 = safe_r(arr1, arr2)
        r23 = safe_r(arr2, arr3)
        r13 = safe_r(arr1, arr3)
        rbar = fisher_mean_r([r12, r23, r13])

        mat = np.vstack([arr1, arr2, arr3]).T
        icc31, icc3k = icc_3(mat)

        rows.append({
            "metric": m,
            "n_models": len(bases),
            "r(run1,run2)": None if np.isnan(r12) else round(r12, 3),
            "r(run2,run3)": None if np.isnan(r23) else round(r23, 3),
            "r(run1,run3)": None if np.isnan(r13) else round(r13, 3),
            "r̄ Fisher-avg": None if np.isnan(rbar) else round(rbar, 3),
            "ICC(3,1)": None if np.isnan(icc31) else round(icc31, 3),
            "ICC(3,k)": None if np.isnan(icc3k) else round(icc3k, 3),
        })

    return pd.DataFrame(rows)


def compute_by_family(bases, complete):
    fam_rows = []
    metrics = ["virtue", "deontology", "consequentialism", "flip_rate"]

    families = sorted(set(complete[b][1]["family"] for b in bases))

    for fam in families:
        fam_bases = [b for b in bases if complete[b][1]["family"] == fam]
        if len(fam_bases) < 2:
            continue

        for m in metrics:
            v1, v2, v3 = [], [], []
            for b in fam_bases:
                if m == "flip_rate":
                    v1.append(complete[b][1]["flip_rate"])
                    v2.append(complete[b][2]["flip_rate"])
                    v3.append(complete[b][3]["flip_rate"])
                else:
                    v1.append(complete[b][1]["weights"][m])
                    v2.append(complete[b][2]["weights"][m])
                    v3.append(complete[b][3]["weights"][m])

            r12 = safe_r(v1, v2)
            r23 = safe_r(v2, v3)
            r13 = safe_r(v1, v3)
            rbar = fisher_mean_r([r12, r23, r13])

            mat = np.vstack([v1, v2, v3]).T
            icc31, icc3k = icc_3(mat)

            fam_rows.append({
                "family": fam,
                "metric": m,
                "n_models": len(fam_bases),
                "r(run1,run2)": None if np.isnan(r12) else round(r12, 3),
                "r(run2,run3)": None if np.isnan(r23) else round(r23, 3),
                "r(run1,run3)": None if np.isnan(r13) else round(r13, 3),
                "r̄ Fisher-avg": None if np.isnan(rbar) else round(rbar, 3),
                "ICC(3,1)": None if np.isnan(icc31) else round(icc31, 3),
                "ICC(3,k)": None if np.isnan(icc3k) else round(icc3k, 3),
            })

    return pd.DataFrame(fam_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", required=True, help="Path to estimates.json")
    ap.add_argument("--outdir", required=True, help="Directory to write CSV outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bases, complete = load_complete_triplets(args.estimates)
    if not bases:
        raise SystemExit("No models found with all three runs (_2/_3 suffix).")

    overall = compute_overall(bases, complete)
    per_family = compute_by_family(bases, complete)

    overall_path = os.path.join(args.outdir, "run_reliability_summary.csv")
    per_family_path = os.path.join(args.outdir, "run_reliability_by_family.csv")
    overall.to_csv(overall_path, index=False)
    per_family.to_csv(per_family_path, index=False)

    print(f"[ok] Wrote: {overall_path}")
    if not per_family.empty:
        print(f"[ok] Wrote: {per_family_path}")
    else:
        print("[note] Not enough models per family to compute per-family reliability.")


if __name__ == "__main__":
    main()
