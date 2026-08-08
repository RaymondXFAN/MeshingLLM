"""
stats_analysis.py — 统计分析（避免"看起来显著其实是噪声"）

读取 outputs/<exp>/results.jsonl（每行一个 {method, dataset, seed, metric:值}），
生成：
  - mean ± std
  - bootstrap 95% CI (1000 resamples)
  - 配对 Wilcoxon / t 检验（MeshingLLM vs 各基线）
  - 多重比较 Holm 校正
  - IEEE 双栏窄表 (markdown / csv)

依赖: pandas, scipy, numpy
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def load(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(df: pd.DataFrame, metric: str) -> dict:
    out = {}
    for method, g in df.groupby("method"):
        vals = g[metric].dropna().values
        if len(vals) == 0:
            continue
        ci = boot_ci(vals)
        out[method] = dict(n=len(vals), mean=float(vals.mean()),
                           std=float(vals.std(ddof=1)),
                           ci_low=float(ci[0]), ci_high=float(ci[1]))
    return out


def compare(df: pd.DataFrame, metric: str, target: str = "MeshingLLM (L2)") -> pd.DataFrame:
    rows = []
    base = df[df.method == target][metric].dropna().values
    for method, g in df.groupby("method"):
        if method == target:
            continue
        comp = g[metric].dropna().values
        if len(base) < 2 or len(comp) < 2:
            continue
        paired = (len(base) == len(comp))
        t, pt = (stats.ttest_rel(base, comp) if paired
                  else stats.ttest_ind(base, comp, equal_var=False))
        w, pw = (stats.wilcoxon(base, comp) if paired else (np.nan, np.nan))
        # 效应量 Cohen's d
        pool_std = np.sqrt((base.var(ddof=1) + comp.var(ddof=1)) / 2)
        d = (base.mean() - comp.mean()) / pool_std if pool_std > 0 else 0.0
        rows.append(dict(compare=f"{target} vs {method}",
                        mean_diff=float(base.mean() - comp.mean()),
                        t=float(t), p_t=float(pt),
                        p_wilcoxon=(float(pw) if not np.isnan(pw) else None),
                        cohens_d=float(d)))
    res = pd.DataFrame(rows)
    if len(res):
        # Holm 逐步校正
        order = res.p_t.argsort().values
        m = len(res)
        res["p_holm"] = 1.0
        prev = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, max(prev, (m - rank) * res.p_t.iloc[idx]))
            res.loc[res.index[idx], "p_holm"] = val
            prev = val
    return res


def to_ieee_table(summary: dict, metric: str, filename: str):
    lines = ["| Method | Mean±Std | 95% CI |", "|---|---|---|"]
    for method, s in summary.items():
        lines.append(
            f"| {method} | {s['mean']:.1f}±{s['std']:.1f} "
            f"| [{s['ci_low']:.1f}, {s['ci_high']:.1f}] |")
    Path(filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"[table] -> {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--target", default="MeshingLLM (L2)")
    ap.add_argument("--outdir", default="outputs/tables")
    a = ap.parse_args()

    df = load(a.results)
    s = summarize(df, a.metric)
    print(f"=== Summary ({a.metric}: mean±std, 95% CI) ===")
    for k, v in s.items():
        print(f"  {k}: {v['mean']:.2f}±{v['std']:.2f}  CI[{v['ci_low']:.2f},{v['ci_high']:.2f}]")

    cmp = compare(df, a.metric, a.target)
    print(f"\n=== Compare vs {a.target} (paired test + Holm) ===")
    print(cmp.to_string(index=False))

    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    to_ieee_table(s, a.metric, f"{a.outdir}/{a.metric}_table.md")
    cmp.to_csv(f"{a.outdir}/{a.metric}_compare.csv", index=False)


if __name__ == "__main__":
    main()
