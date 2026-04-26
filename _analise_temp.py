import pandas as pd
import numpy as np
from scipy import stats

print("=== CARREGANDO DADOS ===")
df = pd.read_parquet(r'C:\Users\Lucas\Desktop\FIFA\escanteios\dados_escanteios\features_ml.parquet')
print(f'Shape: {df.shape}')
print(f'snap_minutes: {sorted(df["snap_minute"].unique())}')

# Verificar colunas chave
cols_check = ['corners_total_so_far', 'target_corners_total', 'expected_remaining_corners',
              'corners_rate_per_min', 'urgency', 'remaining_time', 'event_id',
              'corners_last_5min', 'corners_last_10min', 'delta_corners_total_so_far',
              'corners_over_odds', 'corners_under_odds', 'league_id']
print("\n--- Colunas disponíveis ---")
for c in cols_check:
    avail = c in df.columns
    if avail:
        null_pct = df[c].isna().mean()*100
        print(f"  {c}: OK | null={null_pct:.1f}%")
    else:
        print(f"  {c}: AUSENTE")

# ============================================================
# 1. DISTRIBUIÇÃO DO TARGET POR SNAP_MINUTE
# ============================================================
print("\n\n=== 1. DISTRIBUIÇÃO DO TARGET POR SNAP_MINUTE ===")
g = df.groupby("snap_minute")["target_corners_total"]
stats_df = g.agg(["count", "mean", "std", "min", "max"])
stats_df.columns = ["N", "mean", "std", "min", "max"]
for m, row in stats_df.iterrows():
    print(f"  min{int(m)}: N={int(row['N']):,} | mean={row['mean']:.2f} | std={row['std']:.2f} | range=[{row['min']:.0f},{row['max']:.0f}]")

# ============================================================
# 2. CALIBRAÇÃO SIMULADA: PREDITOR SIMPLES
# ============================================================
print("\n\n=== 2. PREDITOR SIMPLES: corners_so_far + expected_remaining ===")
has_remaining = 'expected_remaining_corners' in df.columns

rows = []
for m in [15, 30, 45, 60, 75]:
    sub = df[df["snap_minute"] == m].copy()

    # Preditor B1: extrapolação linear
    B1 = sub["corners_total_so_far"] / m * 90

    # Preditor simples (se disponível)
    if has_remaining:
        pred_simple = sub["corners_total_so_far"] + sub["expected_remaining_corners"].fillna(
            (90 - m) * sub["corners_rate_per_min"].fillna(0.1))
    else:
        pred_simple = sub["corners_total_so_far"] + (90 - m) * sub["corners_rate_per_min"].fillna(0.1)

    actual = sub["target_corners_total"]

    # Métricas
    mae_B1 = (B1 - actual).abs().mean()
    mae_simple = (pred_simple - actual).abs().mean()
    bias_B1 = (B1 - actual).mean()
    bias_simple = (pred_simple - actual).mean()
    rmse_simple = np.sqrt(((pred_simple - actual)**2).mean())

    # Percentis do erro
    err = (pred_simple - actual)
    p10, p25, p50, p75, p90 = np.percentile(err.dropna(), [10, 25, 50, 75, 90])

    # Over-smoothing: variância
    if has_remaining:
        remaining_pred = sub["expected_remaining_corners"].dropna()
        remaining_actual = (actual - sub["corners_total_so_far"]).reindex(remaining_pred.index)
        var_ratio = remaining_pred.var() / max(remaining_actual.var(), 0.001)
    else:
        var_ratio = np.nan

    rows.append({
        "min": m, "mae_B1": mae_B1, "mae_simple": mae_simple,
        "bias_B1": bias_B1, "bias_simple": bias_simple, "rmse_simple": rmse_simple,
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "var_ratio_remaining": var_ratio,
        "err_gt3_pct": (err.abs() > 3).mean() * 100
    })

    gain = (mae_B1 - mae_simple) / mae_B1 * 100
    print(f"  min{m}: MAE_B1={mae_B1:.3f} | MAE_simple={mae_simple:.3f} | ganho={gain:+.1f}% | bias={bias_simple:+.3f} | RMSE={rmse_simple:.3f}")
    print(f"         err percentis: [{p10:.1f},{p25:.1f},{p50:.1f},{p75:.1f},{p90:.1f}] | err>3: {(err.abs()>3).mean()*100:.1f}%")
    if has_remaining:
        print(f"         var_pred_remaining/var_actual_remaining = {var_ratio:.3f} {'(OVER-SMOOTH)' if var_ratio < 0.7 else ''}")

# ============================================================
# 3. VIÉS POR QUANTIDADE DE CORNERS ACUMULADOS (min45, min60)
# ============================================================
print("\n\n=== 3. VIÉS POR CORNERS ACUMULADOS (min45 e min60) ===")
for m in [45, 60]:
    sub = df[df["snap_minute"] == m].copy()
    if has_remaining:
        pred = sub["corners_total_so_far"] + sub["expected_remaining_corners"].fillna(0)
    else:
        pred = sub["corners_total_so_far"] + (90 - m) * sub["corners_rate_per_min"].fillna(0.1)

    actual = sub["target_corners_total"]
    err = (pred - actual)
    sub["err"] = err
    sub["so_far"] = sub["corners_total_so_far"]

    faixas = [(0, 4, "frio (0-4)"), (4, 7, "normal (5-7)"), (7, 10, "quente (8-10)"), (10, 99, "muito quente (11+)")]
    print(f"\n  min{m}:")
    for lo, hi, label in faixas:
        mask = (sub["so_far"] >= lo) & (sub["so_far"] < hi)
        n = mask.sum()
        if n > 0:
            b = sub.loc[mask, "err"].mean()
            std_e = sub.loc[mask, "err"].std()
            print(f"    {label:25s}: N={n:,} | bias={b:+.2f} | std_err={std_e:.2f}")

# ============================================================
# 4. CORRELAÇÕES CHAVE
# ============================================================
print("\n\n=== 4. CORRELAÇÕES COM TARGET E ENTRE FEATURES ===")
feat_corr = ['corners_total_so_far', 'corners_rate_per_min', 'expected_remaining_corners',
             'urgency', 'remaining_time', 'corners_last_5min', 'corners_last_10min']
feat_corr = [f for f in feat_corr if f in df.columns]

print(f"\n  {'Feature':<35s} {'corr(feat,target)':>18s}  {'corr(feat,so_far)':>18s}")
for m in [45, 60]:
    sub = df[df["snap_minute"] == m].dropna(subset=["target_corners_total", "corners_total_so_far"])
    print(f"\n  --- min{m} ---")
    for f in feat_corr:
        if f in sub.columns:
            c_target = sub[f].corr(sub["target_corners_total"])
            c_soFar = sub[f].corr(sub["corners_total_so_far"])
            print(f"  {f:<35s} {c_target:>+18.3f}  {c_soFar:>+18.3f}")

# ============================================================
# 5. ANCORAGEM: CORRELAÇÃO SO_FAR vs TARGET POR MINUTO
# ============================================================
print("\n\n=== 5. ANCORAGEM: corr(corners_so_far, target) por minuto ===")
for m in [15, 30, 45, 60, 75]:
    sub = df[df["snap_minute"] == m].dropna(subset=["target_corners_total", "corners_total_so_far"])
    c = sub["corners_total_so_far"].corr(sub["target_corners_total"])

    # Correlação parcial simples (resíduo de regressão linear)
    from numpy.linalg import lstsq
    X = sub["corners_total_so_far"].values.reshape(-1, 1)
    y = sub["target_corners_total"].values
    coef, _, _, _ = lstsq(np.hstack([X, np.ones_like(X)]), y, rcond=None)

    print(f"  min{m}: corr={c:.4f} | OLS coef so_far={coef[0]:.4f}, intercept={coef[1]:.4f}")

# ============================================================
# 6. ODDS VERIFICAÇÃO
# ============================================================
print("\n\n=== 6. VERIFICAÇÃO DAS ODDS ===")
for col in ['corners_over_odds', 'corners_under_odds']:
    if col in df.columns:
        print(f"\n  {col}:")
        print(f"    null: {df[col].isna().mean()*100:.1f}%")
        non_null = df[col].dropna()
        if len(non_null) > 0:
            print(f"    describe: {non_null.describe().to_dict()}")
            print(f"    value_counts top-5: {non_null.value_counts().head(5).to_dict()}")
    else:
        print(f"\n  {col}: AUSENTE no dataset")

# ============================================================
# 7. ACURÁCIA OVER/UNDER com linha dinâmica simples (@ 1.83)
# ============================================================
print("\n\n=== 7. ACURÁCIA REAL COM LINHA DINÂMICA (break-even 54.64%) ===")
ODDS = 1.83
BE = 1 / ODDS

for m in [15, 30, 45, 60, 75]:
    sub = df[df["snap_minute"] == m].dropna(subset=["target_corners_total", "corners_total_so_far"]).copy()

    # Linha dinâmica: acumulado + projeção linear
    rate = sub["corners_rate_per_min"].fillna(sub["corners_total_so_far"] / m)
    linha = sub["corners_total_so_far"] + (90 - m) * rate

    # Previsão simples
    if has_remaining:
        pred = sub["corners_total_so_far"] + sub["expected_remaining_corners"].fillna((90 - m) * rate)
    else:
        pred = sub["corners_total_so_far"] + (90 - m) * rate

    actual = sub["target_corners_total"]

    # Over se pred > linha
    side_over = (pred > linha)
    won_over = (actual > linha)[side_over].mean() if side_over.sum() > 0 else 0
    n_over = side_over.sum()

    # Under se pred < linha
    side_under = (pred < linha)
    won_under = (actual <= linha)[side_under].mean() if side_under.sum() > 0 else 0
    n_under = side_under.sum()

    roi_over = (won_over * (ODDS - 1) - (1 - won_over)) if n_over > 0 else 0
    roi_under = (won_under * (ODDS - 1) - (1 - won_under)) if n_under > 0 else 0

    print(f"  min{m}: Over N={n_over:,} acc={won_over:.1%} ROI={roi_over:+.1%} | Under N={n_under:,} acc={won_under:.1%} ROI={roi_under:+.1%}")

# ============================================================
# 8. DISTRIBUIÇÃO DE CORNERS: % jogos acima de linhas
# ============================================================
print("\n\n=== 8. DISTRIBUIÇÃO DE CORNERS FINAIS ===")
pan = pd.read_parquet(r'C:\Users\Lucas\Desktop\FIFA\escanteios\dados_escanteios\panorama_jogos.parquet')
print(f"  panorama_jogos: {pan.shape}")
if 'corners_total' in pan.columns:
    ct = pan['corners_total'].dropna()
    for linha in [8.5, 9.0, 9.5, 10.0, 10.5, 11.0]:
        pct = (ct > linha).mean()
        print(f"  Over {linha}: {pct:.1%} ({int(ct.shape[0]*pct):,} de {ct.shape[0]:,} jogos)")

# ============================================================
# 9. AUTOCORRELAÇÃO DE RESÍDUOS INTRA-JOGO
# ============================================================
print("\n\n=== 9. AUTOCORRELAÇÃO DE RESÍDUOS INTRA-JOGO ===")
if 'event_id' in df.columns and has_remaining:
    df2 = df[df["snap_minute"].isin([15, 30, 45, 60, 75])].copy()
    df2["pred_simple"] = df2["corners_total_so_far"] + df2["expected_remaining_corners"].fillna(0)
    df2["residual"] = df2["pred_simple"] - df2["target_corners_total"]

    pivot = df2.pivot_table(index="event_id", columns="snap_minute", values="residual")

    for (m1, m2) in [(15, 30), (30, 45), (45, 60), (60, 75)]:
        if m1 in pivot.columns and m2 in pivot.columns:
            mask = pivot[[m1, m2]].notna().all(axis=1)
            n_pairs = mask.sum()
            if n_pairs > 100:
                corr = pivot.loc[mask, m1].corr(pivot.loc[mask, m2])
                print(f"  Autocorr min{m1}→min{m2}: {corr:.4f} (N={n_pairs:,}) {'⚠ ALTO' if corr > 0.4 else 'OK'}")
else:
    print("  event_id ou expected_remaining ausentes — skip")

# ============================================================
# 10. DISTRIBUIÇÃO LogNormal vs NegBinom
# ============================================================
print("\n\n=== 10. TESTE DE DISTRIBUIÇÃO DO TARGET (min45) ===")
sub45 = df[df["snap_minute"] == 45]["target_corners_total"].dropna()
mu = sub45.mean()
var = sub45.var()
print(f"  mean={mu:.3f} | var={var:.3f} | overdispersion ratio={var/mu:.3f}")

# Poisson KS
ks_p, pv_p = stats.kstest(sub45, lambda x: stats.poisson.cdf(x, mu))
print(f"  KS vs Poisson:        stat={ks_p:.4f}, p={pv_p:.2e}")

# NegBinom KS
if var > mu:
    p_nb = mu**2 / (var - mu)
    r_nb = mu * p_nb / (1 - p_nb)
    ks_nb, pv_nb = stats.kstest(sub45, lambda x: stats.nbinom.cdf(x, r_nb, p_nb))
    print(f"  KS vs NegativeBinom:  stat={ks_nb:.4f}, p={pv_nb:.2e}")

# Normal
ks_n, pv_n = stats.kstest(sub45, lambda x: stats.norm.cdf(x, mu, sub45.std()))
print(f"  KS vs Normal:         stat={ks_n:.4f}, p={pv_n:.2e}")

print("\n=== ANÁLISE CONCLUÍDA ===")
