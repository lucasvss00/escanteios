# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_parquet(r'C:\Users\Lucas\Desktop\FIFA\escanteios\dados_escanteios\features_ml.parquet')
has_remaining = 'expected_remaining_corners' in df.columns

# ============================================================
# 9. AUTOCORRELAÇÃO DE RESÍDUOS INTRA-JOGO
# ============================================================
print("=== 9. AUTOCORRELACAO DE RESIDUOS INTRA-JOGO ===")
if 'event_id' in df.columns and has_remaining:
    df2 = df[df["snap_minute"].isin([15, 30, 45, 60, 75])].copy()
    df2["pred_simple"] = df2["corners_total_so_far"] + df2["expected_remaining_corners"]
    df2["residual"] = df2["pred_simple"] - df2["target_corners_total"]

    pivot = df2.pivot_table(index="event_id", columns="snap_minute", values="residual")

    for (m1, m2) in [(15, 30), (30, 45), (45, 60), (60, 75)]:
        if m1 in pivot.columns and m2 in pivot.columns:
            mask = pivot[[m1, m2]].notna().all(axis=1)
            n_pairs = mask.sum()
            if n_pairs > 100:
                corr = pivot.loc[mask, m1].corr(pivot.loc[mask, m2])
                flag = "ALTO - lag significativo!" if corr > 0.4 else "OK"
                print(f"  Autocorr min{m1}->min{m2}: {corr:.4f} (N={n_pairs:,}) [{flag}]")

# ============================================================
# 10. DISTRIBUIÇÃO DO TARGET: KS TEST
# ============================================================
print("\n=== 10. TESTE DE DISTRIBUICAO DO TARGET ===")
for m in [15, 30, 45, 60, 75]:
    sub = df[df["snap_minute"] == m]["target_corners_total"].dropna()
    mu = sub.mean()
    var = sub.var()
    od = var / mu
    print(f"\n  min{m}: mean={mu:.3f} | var={var:.3f} | overdispersion={od:.3f}")

    # KS Poisson
    ks_p, pv_p = stats.kstest(sub, lambda x: stats.poisson.cdf(x, mu))
    print(f"    KS Poisson:       stat={ks_p:.4f}, p={pv_p:.2e}")

    # KS NegBinom
    if var > mu:
        p_nb = mu**2 / (var - mu)
        r_nb = mu * p_nb / (1 - p_nb)
        ks_nb, pv_nb = stats.kstest(sub, lambda x: stats.nbinom.cdf(x, r_nb, p_nb))
        print(f"    KS NegBinom:      stat={ks_nb:.4f}, p={pv_nb:.2e}")

    # KS Normal
    ks_n, pv_n = stats.kstest(sub, lambda x: stats.norm.cdf(x, mu, sub.std()))
    print(f"    KS Normal:        stat={ks_n:.4f}, p={pv_n:.2e}")

# ============================================================
# 11. COLINEARIDADE: corr entre top features
# ============================================================
print("\n=== 11. COLINEARIDADE DAS TOP FEATURES (min45) ===")
sub45 = df[df["snap_minute"] == 45]
top_feats = ["corners_total_so_far", "corners_rate_per_min", "expected_remaining_corners",
             "corners_last_5min", "corners_last_10min", "urgency",
             "league_id_target_enc", "da_acceleration_vs_global"]
top_feats = [f for f in top_feats if f in sub45.columns]

print("  Correlacao entre features top (min45):")
print(f"  {'':35s}", end="")
for f in top_feats:
    print(f"  {f[:12]:>12s}", end="")
print()
for f1 in top_feats:
    print(f"  {f1:35s}", end="")
    for f2 in top_feats:
        c = sub45[f1].corr(sub45[f2])
        marker = "****" if abs(c) > 0.95 and f1 != f2 else f"{c:+.2f}"
        print(f"  {marker:>12s}", end="")
    print()

# ============================================================
# 12. IMPACTO DAS FEATURES MOMENTÂNEAS NA PREVISAO
# ============================================================
print("\n=== 12. FEATURES DE MOMENTUM vs RESIDUO (min45 e min60) ===")
for m in [30, 45, 60, 75]:
    sub = df[df["snap_minute"] == m].copy()
    sub["pred_simple"] = sub["corners_total_so_far"] + sub["expected_remaining_corners"]
    sub["residual"] = sub["pred_simple"] - sub["target_corners_total"]

    momentum_cols = [c for c in ["corners_last_5min", "corners_last_10min",
                                  "delta_corners_total_so_far",
                                  "attacks_home", "attacks_away",
                                  "dangerous_attacks_home", "dangerous_attacks_away"] if c in sub.columns]

    print(f"\n  min{m}:")
    corrs = {}
    for c in momentum_cols:
        corr_res = sub[c].corr(sub["residual"])
        corr_soFar = sub[c].corr(sub["corners_total_so_far"])
        corrs[c] = (corr_res, corr_soFar)

    for c, (cr, cs) in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True):
        print(f"    {c:40s}  corr_residuo={cr:+.3f}  corr_so_far={cs:+.3f}")

print("\n=== FIM ===")
