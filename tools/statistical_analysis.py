
import argparse
import json
import math
import os
import itertools
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple
from scipy.stats import chi2_contingency, norm, ttest_ind


# --- Utilities ---

def bh_fdr(pvals: List[float], alpha: float = 0.05) -> List[float]:
    if not pvals:
        return []

    cleaned: List[float] = []
    for p in pvals:
        try:
            val = float(p)
        except (TypeError, ValueError):
            val = math.nan
        cleaned.append(val if not math.isnan(val) else math.nan)

    arr = np.asarray(cleaned, dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any():
        return arr.tolist()

    adj = _bh_adjust(arr[mask])
    q = np.full_like(arr, math.nan, dtype=float)
    q[mask] = np.minimum(1.0, adj)
    return q.tolist()

def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        cummin = min(cummin, val)
        q[i] = cummin
    out = np.empty_like(q)
    out[order] = q
    return out


def cramers_v(chi2: float, n_total: int, r: int, c: int) -> float:
    if n_total <= 0 or min(r, c) <= 1:
        return math.nan
    return math.sqrt(chi2 / (n_total * (min(r - 1, c - 1))))


def twoproportion_z(s1: int, n1: int, s2: int, n2: int) -> Tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (math.nan, math.nan)
    p1 = s1 / n1
    p2 = s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return (math.nan, math.nan)
    z = (p1 - p2) / denom
    p = 2 * norm.sf(abs(z))
    return (z, p)

def _extract_model_axis_df(models_or_df) -> pd.DataFrame:
    if isinstance(models_or_df, pd.DataFrame):
        cols = {"model","family","V","D","C"}
        if cols.issubset(set(models_or_df.columns)):
            return models_or_df[list(cols)].copy()

    data = models_or_df
    if isinstance(data, dict):
        if "models" in data and isinstance(data["models"], list):
            data = data["models"]
        elif "models_raw" in data and isinstance(data["models_raw"], list):
            data = data["models_raw"]
        else:
            data = []

    rows = []
    for m in data:
        name = m.get("label") or m.get("model") or m.get("name")
        fam  = m.get("family") or m.get("Family") or m.get("group")

        V = D = C = None

        fw = m.get("final_weights")
        if isinstance(fw, dict):
            V = fw.get("virtue")
            D = fw.get("deontology")
            C = fw.get("consequentialism")

        if any(x is None for x in (V, D, C)):
            w = m.get("weights")
            if isinstance(w, dict):
                V = w.get("virtue", V)
                D = w.get("deontology", D)
                C = w.get("consequentialism", C)

        if any(x is None for x in (V, D, C)):
            cnt = m.get("counts") or {}
            nA = cnt.get("A", 0)
            nB = cnt.get("B", 0)
            nC = cnt.get("C", 0)
            N  = nA + nB + nC
            if N > 0:
                V = nA / N if V is None else V
                D = nB / N if D is None else D
                C = nC / N if C is None else C

        rows.append({"model": name, "family": fam, "V": V, "D": D, "C": C})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["family", "V", "D", "C"]).reset_index(drop=True)
    return df


# --- Data loading ---

def load_models(estimates_path: str) -> List[Dict]:
    with open(estimates_path, "r") as f:
        data = json.load(f)
    models = data["models"]
    return models


def family_groups(models: List[Dict]) -> Dict[str, List[Dict]]:
    fam = {}
    for m in models:
        fam.setdefault(m["family"], []).append(m)
    for k in fam:
        fam[k] = sorted(fam[k], key=lambda x: x["label"])
    return fam


# --- Omnibus weights (within-family) ---

def omnibus_within_family_weights(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for fam, models in fam_models.items():
        k = len(models)
        A = [m["counts"]["A"] for m in models]
        B = [m["counts"]["B"] for m in models]
        C = [m["counts"]["C"] for m in models]
        table = np.array([A, B, C], dtype=float)
        N = int(table.sum())
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        V = cramers_v(chi2, N, r=3, c=k)
        rows.append({
            "family": fam,
            "k_models": k,
            "N_total": N,
            "chi2": chi2,
            "dof": dof,
            "p": p,
            "cramer_v": V
        })
    return pd.DataFrame(rows)

# --- Omnibus flip-rates (within-family) ---

def omnibus_within_family_fliprate(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for fam, models in fam_models.items():
        k = len(models)
        if k < 2:
            continue

        switch = [m["counts"]["SWITCH"] for m in models]
        keep   = [m["counts"]["KEEP"]   for m in models]

        table = np.array([switch, keep], dtype=float)
        N = int(table.sum())

        chi2, p, dof, _ = chi2_contingency(table, correction=False)

        V = cramers_v(chi2, N, r=2, c=k)

        rows.append({
            "family": fam,
            "k_models": k,
            "N_total": N,
            "chi2": chi2,
            "dof": dof,
            "p": p,
            "cramer_v": V
        })

    return pd.DataFrame(rows)


# --- Pairwise within-family weights ---

def pairwise_within_family_weights(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    recs = []
    for fam, models in fam_models.items():
        if len(models) < 2:
            continue
        for m1, m2 in combinations(models, 2):
            a1, b1, c1 = m1["counts"]["A"], m1["counts"]["B"], m1["counts"]["C"]
            a2, b2, c2 = m2["counts"]["A"], m2["counts"]["B"], m2["counts"]["C"]
            table = np.array([[a1, a2], [b1, b2], [c1, c2]], dtype=float)
            N = int(table.sum())
            chi2, p, dof, _ = chi2_contingency(table, correction=False)
            V = cramers_v(chi2, N, r=3, c=2)
            recs.append({
                "family": fam,
                "model_A": m1["label"],
                "model_B": m2["label"],
                "A_counts": f"{a1}:{a2}",
                "B_counts": f"{b1}:{b2}",
                "C_counts": f"{c1}:{c2}",
                "N_total": N,
                "chi2": chi2,
                "dof": dof,
                "p": p,
                "cramer_v": V
            })
    df = pd.DataFrame(recs)
    if not df.empty:
        df["q"] = bh_fdr(df["p"].tolist(), alpha=0.05)
    return df


# --- Pairwise within-family flip-rates ---

def pairwise_within_family_flips(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    recs = []
    for fam, models in fam_models.items():
        if len(models) < 2:
            continue
        for m1, m2 in combinations(models, 2):
            sw1, keep1 = m1["counts"]["SWITCH"], m1["counts"]["KEEP"]
            sw2, keep2 = m2["counts"]["SWITCH"], m2["counts"]["KEEP"]
            n1, n2 = sw1 + keep1, sw2 + keep2
            z, p = twoproportion_z(sw1, n1, sw2, n2)
            p1 = sw1 / n1 if n1 else math.nan
            p2 = sw2 / n2 if n2 else math.nan
            recs.append({
                "family": fam,
                "model_A": m1["label"],
                "model_B": m2["label"],
                "switch_A": sw1,
                "total_A": n1,
                "switch_B": sw2,
                "total_B": n2,
                "phi_A": p1,
                "phi_B": p2,
                "diff_phi": (p1 - p2) if not (math.isnan(p1) or math.isnan(p2)) else math.nan,
                "z": z,
                "p": p
            })
    df = pd.DataFrame(recs)
    if not df.empty:
        df["q"] = bh_fdr(df["p"].tolist(), alpha=0.05)
    return df


# --- Between-family weights (pooled) ---

def between_family_weights(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    pooled = {}
    for fam, models in fam_models.items():
        A = sum(m["counts"]["A"] for m in models)
        B = sum(m["counts"]["B"] for m in models)
        C = sum(m["counts"]["C"] for m in models)
        pooled[fam] = (A, B, C)

    recs = []
    fam_list = sorted(pooled.keys())
    for f1, f2 in combinations(fam_list, 2):
        a1, b1, c1 = pooled[f1]
        a2, b2, c2 = pooled[f2]
        table = np.array([[a1, a2], [b1, b2], [c1, c2]], dtype=float)
        N = int(table.sum())
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        V = cramers_v(chi2, N, r=3, c=2)
        recs.append({
            "family_A": f1,
            "family_B": f2,
            "A_counts": f"{a1}:{a2}",
            "B_counts": f"{b1}:{b2}",
            "C_counts": f"{c1}:{c2}",
            "N_total": N,
            "chi2": chi2,
            "dof": dof,
            "p": p,
            "cramer_v": V
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        df["q"] = bh_fdr(df["p"].tolist(), alpha=0.05)
    return df


# --- Between-family flips (pooled) ---

def between_family_flips(fam_models: Dict[str, List[Dict]]) -> pd.DataFrame:
    pooled = {}
    for fam, models in fam_models.items():
        sw = sum(m["counts"]["SWITCH"] for m in models)
        keep = sum(m["counts"]["KEEP"] for m in models)
        pooled[fam] = (sw, keep)

    recs = []
    fam_list = sorted(pooled.keys())
    for f1, f2 in combinations(fam_list, 2):
        sw1, keep1 = pooled[f1]; n1 = sw1 + keep1
        sw2, keep2 = pooled[f2]; n2 = sw2 + keep2
        z, p = twoproportion_z(sw1, n1, sw2, n2)
        phi1 = sw1 / n1 if n1 else math.nan
        phi2 = sw2 / n2 if n2 else math.nan
        recs.append({
            "family_A": f1,
            "family_B": f2,
            "switch_A": sw1,
            "total_A": n1,
            "switch_B": sw2,
            "total_B": n2,
            "phi_A": phi1,
            "phi_B": phi2,
            "diff_phi": (phi1 - phi2) if not (math.isnan(phi1) or math.isnan(phi2)) else math.nan,
            "z": z,
            "p": p
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        df["q"] = bh_fdr(df["p"].tolist(), alpha=0.05)
    return df

# --- Between-family weight means ---

def between_family_weights_axis_means(models_or_df,
                                      out_csv: str,
                                      B: int = 10_000,
                                      seed: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    df = _extract_model_axis_df(models_or_df)
    if df.empty:
        out = pd.DataFrame(columns=[
            "axis","fam1","fam2","n1","n2","mean1","mean2","diff",
            "ci_lo","ci_hi","t","df","p","q","cohen_d"
        ])
        out.to_csv(out_csv, index=False)
        return out

    df["family"] = df["family"].astype(str).str.strip()

    axes = ["V", "D", "C"]
    fams = sorted(df["family"].dropna().unique().tolist())
    tests = []

    def _cohen_d(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return np.nan
        s1, s2 = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
        sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
        return (np.nanmean(a) - np.nanmean(b)) / sp if sp > 0 else np.nan

    def _welch_t(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        n1, n2 = np.sum(~np.isnan(a)), np.sum(~np.isnan(b))
        v1, v2 = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
        if n1 < 2 or n2 < 2 or v1 < 0 or v2 < 0:
            return float("nan"), float("nan"), float("nan")
        num = (v1/n1 + v2/n2)**2
        den = ((v1**2)/((n1**2)*(n1-1))) + ((v2**2)/((n2**2)*(n2-1)))
        df = num/den if den > 0 else (n1 + n2 - 2)
        return float(t), float(df), float(p)

    def _boot_ci_mean_diff(a, b, B=10_000):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan
        ia = rng.integers(0, len(a), size=(B, len(a)))
        ib = rng.integers(0, len(b), size=(B, len(b)))
        diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        return float(lo), float(hi)

    for fam1, fam2 in itertools.combinations(fams, 2):
        g1 = df[df["family"] == fam1]
        g2 = df[df["family"] == fam2]
        for axis in axes:
            a = g1[axis].astype(float).to_numpy()
            b = g2[axis].astype(float).to_numpy()
            n1, n2 = np.sum(~np.isnan(a)), np.sum(~np.isnan(b))
            if n1 < 1 or n2 < 1:
                continue
            m1, m2 = float(np.nanmean(a)), float(np.nanmean(b))
            diff = m1 - m2
            t, df_welch, p = _welch_t(a, b)
            d = _cohen_d(a, b)
            ci_lo, ci_hi = _boot_ci_mean_diff(a, b, B=B)
            tests.append({
                "axis": axis, "fam1": fam1, "fam2": fam2,
                "n1": int(n1), "n2": int(n2),
                "mean1": m1, "mean2": m2, "diff": diff,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "t": t, "df": df_welch, "p": float(p),
                "cohen_d": d
            })

    if not tests:
        out = pd.DataFrame(columns=[
            "axis","fam1","fam2","n1","n2","mean1","mean2","diff",
            "ci_lo","ci_hi","t","df","p","q","cohen_d"
        ])
        out.to_csv(out_csv, index=False)
        return out

    pvals = np.array([r["p"] for r in tests], dtype=float)
    qvals = _bh_adjust(pvals)
    for r, q in zip(tests, qvals):
        r["q"] = float(q)

    out_df = pd.DataFrame(tests)
    out_df.to_csv(out_csv, index=False)
    return out_df



# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", required=True, help="Path to estimates.json")
    ap.add_argument("--outdir", required=True, help="Directory to write CSV outputs")
    ap.add_argument("--boot", type=int, default=10000, help="Bootstrap draws for CIs (axis means)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    models = load_models(args.estimates)
    fam = family_groups(models)

    df_omni = omnibus_within_family_weights(fam)
    omni_path = os.path.join(args.outdir, "within_family_omnibus_weights.csv")
    df_omni.to_csv(omni_path, index=False)

    df_omni_flip = omnibus_within_family_fliprate(fam)
    omni_flip_path = os.path.join(args.outdir, "within_family_omnibus_flips.csv")
    df_omni_flip.to_csv(omni_flip_path, index=False)

    df_w_within = pairwise_within_family_weights(fam)
    within_w_path = os.path.join(args.outdir, "within_family_weights.csv")
    df_w_within.to_csv(within_w_path, index=False)

    df_f_within = pairwise_within_family_flips(fam)
    within_f_path = os.path.join(args.outdir, "within_family_flips.csv")
    df_f_within.to_csv(within_f_path, index=False)

    df_w_between = between_family_weights(fam)
    between_w_path = os.path.join(args.outdir, "between_family_weights.csv")
    df_w_between.to_csv(between_w_path, index=False)

    df_f_between = between_family_flips(fam)
    between_f_path = os.path.join(args.outdir, "between_family_flips.csv")
    df_f_between.to_csv(between_f_path, index=False)

    axis_means_path = os.path.join(args.outdir, "between_family_weights_axis.csv")
    between_family_weights_axis_means(models, axis_means_path, B=args.boot, seed=args.seed)


    print(f"[ok] Wrote: {omni_path}")
    print(f"[ok] Wrote: {omni_flip_path}")
    print(f"[ok] Wrote: {within_w_path}")
    print(f"[ok] Wrote: {within_f_path}")
    print(f"[ok] Wrote: {between_w_path}")
    print(f"[ok] Wrote: {between_f_path}")
    print(f"[ok] Wrote: {axis_means_path}")



if __name__ == "__main__":
    main()
