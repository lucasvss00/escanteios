"""
calibration_diagnosis.py
========================
Diagnóstico de calibração do NGBoost (distribuição LogNormal).

NOTA IMPORTANTE sobre a distribuição:
  O pipeline usa NGBRegressor com Dist=LogNormal (scipy convention:
  lognorm(s=sigma, scale=exp(mu))). O usuário menciona NegBinom, mas o
  código usa LogNormal — este script opera sobre LogNormal.
  O P(over linha) no pipeline é calculado via Poisson(mu=E[X]) como
  *aproximação*, não via CDF da LogNormal nativa. Ambas as versões são
  avaliadas aqui.

Uso:
    python calibration_diagnosis.py
    python calibration_diagnosis.py --models-dir dados_escanteios
    python calibration_diagnosis.py --no-plots    # sem matplotlib
    python calibration_diagnosis.py --minutes 30 60 75
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import lognorm as sp_lognorm
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET   = Path("dados_escanteios/features_ml.parquet")
DEFAULT_MODELS    = Path("dados_escanteios")
DEFAULT_PLOTS     = Path("dados_escanteios/plots")
SNAPSHOT_MINUTES  = [15, 30, 45, 60, 75]
TARGET            = "target_corners_total"
N_PIT_BINS        = 20
N_REL_BINS        = 20
ODDS              = 1.83


# ---------------------------------------------------------------------------
# HELPERS — split temporal (igual ao pipeline principal)
# ---------------------------------------------------------------------------
def _split_temporal(df_min: pd.DataFrame):
    n     = len(df_min)
    n_te  = int(n * 0.20)
    n_cal = int((n - n_te) * 0.25)
    n_tr  = n - n_te - n_cal
    return (
        df_min.iloc[:n_tr].copy(),
        df_min.iloc[n_tr:n_tr + n_cal].copy(),
        df_min.iloc[n_tr + n_cal:].copy(),
    )


def _prepare_X(
    df: pd.DataFrame,
    feature_list: list[str],
    medians: dict,
    global_mean: float,
) -> np.ndarray:
    """Aplica mesmo fill do pipeline (fillna 0 / mediana histórica)."""
    df2 = df.copy()
    for c in feature_list:
        if c not in df2.columns:
            df2[c] = 0.0
    # Preenche colunas históricas com mediana do treino
    fill_med  = [c for c in feature_list
                 if c.startswith(("hist_", "league_", "h2h_")) or c.endswith("_target_enc")]
    fill_zero = [c for c in feature_list if c not in fill_med]
    for c in fill_med:
        df2[c] = df2[c].fillna(medians.get(c, global_mean))
    for c in fill_zero:
        df2[c] = df2[c].fillna(0)
    df2[feature_list] = df2[feature_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df2[feature_list].values.astype(float)


def _dynamic_line(df: pd.DataFrame, snap_min: int) -> np.ndarray:
    rem = 90 - snap_min
    csf = (df["corners_total_so_far"].values
           if "corners_total_so_far" in df.columns else np.zeros(len(df)))
    if "league_avg_corners" in df.columns:
        rate = np.where(df["league_avg_corners"].isna(),
                        11.0 / 90.0, df["league_avg_corners"].values / 90.0)
    else:
        rate = np.full(len(df), 11.0 / 90.0)
    return csf + rem * rate


# ---------------------------------------------------------------------------
# LOGNORMAL HELPERS
# ---------------------------------------------------------------------------
def lognorm_params(dist_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extrai (s, scale) dos parâmetros da distribuição NGBoost."""
    s     = np.array(dist_params["s"],     dtype=float)
    scale = np.array(dist_params["scale"], dtype=float)
    s     = np.clip(s, 1e-6, 10.0)
    scale = np.clip(scale, 1e-6, 1e6)
    return s, scale


def lognorm_mean(s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """E[X] = scale * exp(s²/2)."""
    return np.clip(scale * np.exp(s ** 2 / 2), 0.01, 60.0)


def lognorm_cdf(x: np.ndarray, s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """CDF da LogNormal em x (vetorizado)."""
    x_safe = np.maximum(x, 1e-9)
    return sp_lognorm.cdf(x_safe, s=s, scale=scale)


def p_over_lognorm(line: np.ndarray, s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """P(Y > line) usando CDF nativa da LogNormal."""
    # Para target discreto: P(Y > floor(line)) = 1 - F(floor(line) + 0.5)
    fl = np.floor(line).astype(float) + 0.5
    return 1.0 - lognorm_cdf(fl, s, scale)


def p_over_poisson_approx(line: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """P(Y > line) via Poisson(mu=E[X]) — como o pipeline atual faz."""
    fl = np.floor(line).astype(int)
    return np.array([1.0 - sp_poisson.cdf(int(fl[i]), mu=max(float(mu[i]), 0.01))
                     for i in range(len(mu))])


# ---------------------------------------------------------------------------
# PARTE A — PIT (Probability Integral Transform)
# ---------------------------------------------------------------------------
def compute_pit(
    y_true: np.ndarray,
    s: np.ndarray,
    scale: np.ndarray,
    randomized: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    PIT randomizado para distribuição LogNormal sobre contagens discretas.

    PIT_i = F(y_i - 1) + U_i * [F(y_i) - F(y_i - 1)], U_i ~ U(0,1)
    Se randomized=False usa PIT contínuo: F(y_i).
    """
    rng = np.random.default_rng(seed)
    y = y_true.astype(float)

    F_y   = lognorm_cdf(y,             s, scale)
    F_ym1 = lognorm_cdf(np.maximum(y - 1.0, 0.0), s, scale)

    if randomized:
        U = rng.uniform(0.0, 1.0, size=len(y))
        pit = F_ym1 + U * (F_y - F_ym1)
    else:
        pit = F_y

    return np.clip(pit, 0.0, 1.0)


def analyse_pit(pit: np.ndarray, snap_min: int) -> dict:
    """KS test + shape metrics para o vetor PIT."""
    ks_stat, ks_pval = sp_stats.kstest(pit, "uniform")

    # Histograma normalizado
    counts, edges = np.histogram(pit, bins=N_PIT_BINS, range=(0, 1), density=False)
    freqs = counts / len(pit)
    expected = 1.0 / N_PIT_BINS

    # Shape: U-invertido (overdispersion) vs U-normal (underdispersion)
    mid = N_PIT_BINS // 2
    center_mass = freqs[mid - 2: mid + 2].mean()
    tail_mass   = np.concatenate([freqs[:3], freqs[-3:]]).mean()
    shape = "overdispersion (U invertido)" if center_mass > expected * 1.15 else (
            "underdispersion (caudas pesadas)" if tail_mass > expected * 1.15 else "OK")

    # Viés: massa acima vs abaixo de 0.5
    bias_left  = (pit < 0.5).mean()   # mais massa na esquerda → superestima Y
    bias_right = (pit > 0.5).mean()   # mais massa na direita → subestima Y
    if bias_left > 0.55:
        bias_dir = f"VIÉS ESQUERDA (pit<0.5={bias_left:.2%}) → modelo SUPERestima corners"
    elif bias_right > 0.55:
        bias_dir = f"VIÉS DIREITA (pit>0.5={bias_right:.2%}) → modelo SUBEstima corners"
    else:
        bias_dir = "sem viés sistemático"

    calibrated = ks_pval >= 0.05
    return {
        "snap_min":  snap_min,
        "n_obs":     len(pit),
        "ks_stat":   round(ks_stat, 4),
        "ks_pval":   round(ks_pval, 6),
        "calibrated": calibrated,
        "shape":     shape,
        "bias":      bias_dir,
        "pit_mean":  round(float(pit.mean()), 4),
        "pit_std":   round(float(pit.std()),  4),
        "freqs":     freqs.tolist(),
        "edges":     edges.tolist(),
    }


# ---------------------------------------------------------------------------
# PARTE B — Reliability diagram + ECE
# ---------------------------------------------------------------------------
def compute_reliability(
    p_pred: np.ndarray,
    y_binary: np.ndarray,
    n_bins: int = N_REL_BINS,
) -> dict:
    """
    Reliability diagram data + ECE.

    p_pred   : probabilidade prevista P(over)
    y_binary : 1 se o jogo realmente passou a linha, 0 caso contrário
    """
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_freqs   = []
    bin_counts  = []
    bin_pred_avg = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        bin_counts.append(int(mask.sum()))
        bin_pred_avg.append(float(p_pred[mask].mean()))
        bin_freqs.append(float(y_binary[mask].mean()))
        bin_centers.append((lo + hi) / 2)

    if not bin_counts:
        return {"ece": float("nan"), "bins": []}

    total    = sum(bin_counts)
    ece      = sum(abs(f - p) * n for f, p, n in zip(bin_freqs, bin_pred_avg, bin_counts)) / total
    overconf = sum((f - p) * n for f, p, n in zip(bin_freqs, bin_pred_avg, bin_counts)) / total

    return {
        "ece":        round(ece, 5),
        "overconf":   round(overconf, 5),   # >0 = underconfident, <0 = overconfident
        "n_total":    total,
        "bin_pred":   bin_pred_avg,
        "bin_freq":   bin_freqs,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def _make_plots(
    pit_result: dict,
    rel_lognorm: dict,
    rel_poisson: dict,
    snap_min: int,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── PIT histogram ──
    fig, ax = plt.subplots(figsize=(8, 4))
    edges = pit_result["edges"]
    freqs = pit_result["freqs"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(freqs))]
    ax.bar(centers, freqs, width=1 / N_PIT_BINS, alpha=0.7,
           color="steelblue", edgecolor="white", label="PIT empírico")
    ax.axhline(1 / N_PIT_BINS, color="red", linestyle="--", lw=1.5, label="Uniforme esperado")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Frequência relativa")
    ks_pval = pit_result["ks_pval"]
    status  = "✅ calibrado" if pit_result["calibrated"] else "❌ descalibrado"
    ax.set_title(f"PIT — NGBoost LogNormal  |  min={snap_min}  |  "
                 f"KS p={ks_pval:.4f} ({status})\n"
                 f"{pit_result['shape']}  |  {pit_result['bias']}", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = plots_dir / f"pit_min{snap_min}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"      Salvo: {out}")

    # ── Reliability diagram ──
    for label, rel, color in [
        ("LogNormal nativa", rel_lognorm, "steelblue"),
        ("Poisson approx (atual)", rel_poisson, "tomato"),
    ]:
        if not rel.get("bin_pred"):
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibração perfeita")
        ax.scatter(rel["bin_pred"], rel["bin_freq"],
                   s=[c * 0.3 for c in rel["bin_counts"]],
                   color=color, alpha=0.8, label=label, zorder=5)
        ax.plot(rel["bin_pred"], rel["bin_freq"], color=color, alpha=0.5)
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Frequência empírica")
        tag = "lognorm" if "Log" in label else "poisson"
        ax.set_title(f"Reliability — {label}\n"
                     f"min={snap_min}  |  ECE={rel['ece']:.4f}  |  "
                     f"overconf={rel['overconf']:+.4f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        out = plots_dir / f"reliability_min{snap_min}_{tag}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"      Salvo: {out}")


# ---------------------------------------------------------------------------
# PARTE C — Diagnóstico numérico + conclusão
# ---------------------------------------------------------------------------
def _overdispersion_ratio(y_true: np.ndarray, mu: np.ndarray) -> float:
    """var(y - ŷ) / mean(ŷ) — >>1 indica underdispersion da distribuição."""
    resid = y_true.astype(float) - mu.astype(float)
    return float(np.var(resid) / max(np.mean(mu), 0.01))


def print_diagnostics_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  TABELA DE DIAGNÓSTICO — NGBoost LogNormal")
    print("=" * 100)
    cols = ["snap_min", "n_obs", "ks_stat", "ks_pval", "calibrated",
            "ece_lognorm", "ece_poisson",
            "bias_mean", "overdispersion", "shape", "bias"]
    avail = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", 20, "display.width", 160,
                           "display.float_format", "{:.4f}".format):
        print(df[avail].to_string(index=False))


def _interpret(row: dict) -> str:
    parts = []
    pval  = row.get("ks_pval", 1.0)
    ece_l = row.get("ece_lognorm", 0.0)
    ece_p = row.get("ece_poisson", 0.0)
    od    = row.get("overdispersion", 1.0)
    m     = row.get("snap_min", "?")

    if pval < 0.01:
        parts.append(f"PIT fortemente não-uniforme (p={pval:.4f})")
    elif pval < 0.05:
        parts.append(f"PIT marginalmente não-uniforme (p={pval:.4f})")
    else:
        parts.append(f"PIT compatível com uniforme (p={pval:.4f}) ✅")

    if ece_l < 0.03:
        parts.append(f"ECE LogNorm={ece_l:.4f} — bem calibrado")
    elif ece_l < 0.06:
        parts.append(f"ECE LogNorm={ece_l:.4f} — calibração moderada")
    else:
        parts.append(f"ECE LogNorm={ece_l:.4f} — DESCALIBRADO → recalibrar")

    if ece_p > ece_l * 1.3:
        parts.append(f"ECE Poisson={ece_p:.4f} pior que LogNorm ({(ece_p/max(ece_l,1e-6)-1)*100:.0f}%) "
                     "→ usar CDF nativa melhora calibração")

    if od > 3:
        parts.append(f"overdispersion={od:.2f} >> 1 — distribuição prevista MUITO estreita "
                     "(underdispersion). Considerar NGBoost com Skellam ou NegBinom.")
    elif od < 0.5:
        parts.append(f"overdispersion={od:.2f} << 1 — distribuição MUITO larga (overdispersion).")

    bias = row.get("bias_mean", 0.0)
    if abs(bias) > 0.5:
        parts.append(f"Viés pontual={bias:+.3f} → {'SUPERestima' if bias > 0 else 'SUBEstima'} corners")

    # Recomendação
    if ece_l > 0.06 or pval < 0.05:
        rec = "AÇÃO: executar recalibrate_ngboost.py --walk-forward"
    elif ece_p > ece_l * 1.3:
        rec = "AÇÃO: substituir Poisson approx por CDF LogNormal nativa (snippet abaixo)"
    else:
        rec = "Sem ação urgente — monitorar MAE e CRPS periodicamente"

    return f"  min{m}: " + " | ".join(parts) + f"\n    → {rec}"


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run(args) -> None:
    try:
        import joblib
    except ImportError:
        print("[ERRO] joblib necessário: pip install joblib")
        sys.exit(1)

    parquet_path = Path(args.parquet)
    models_dir   = Path(args.models_dir)
    plots_dir    = Path(args.plots_dir)

    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}")
        sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df_all = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df_all):,} linhas × {df_all.shape[1]} colunas")

    # Carrega metadata (lista de features por minuto)
    meta_path = models_dir / "modelo_corners_meta.joblib"
    meta = joblib.load(meta_path) if meta_path.exists() else {}
    if not meta:
        print("[AVISO] metadata não encontrado — features serão inferidas automaticamente")

    diag_rows: list[dict] = []
    conclusions: list[str] = []

    minutes = args.minutes or SNAPSHOT_MINUTES

    for snap_min in minutes:
        print(f"\n{'═'*70}")
        print(f"  Minuto {snap_min}")
        print(f"{'═'*70}")

        ngb_path = models_dir / f"modelo_corners_ngb_min{snap_min}.joblib"
        if not ngb_path.exists():
            print(f"  [AVISO] Modelo NGBoost não encontrado: {ngb_path}. Pulando.")
            continue

        ngb_model = joblib.load(ngb_path)
        print(f"  NGBoost carregado: {ngb_path}")

        # ── Reconstrói split temporal ──
        df_min = df_all[df_all["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
        df_tr, df_ca, df_te = _split_temporal(df_min)

        # ── Aplica target encoding do treino (se disponível) ──
        te_path = models_dir / f"target_encoder_min{snap_min}.joblib"
        if te_path.exists():
            te = joblib.load(te_path)
            df_tr = te.transform(df_tr)
            df_ca = te.transform(df_ca)
            df_te = te.transform(df_te)
            print(f"  Target encoder aplicado ({te_path.name})")
        else:
            print(f"  [AVISO] target_encoder_min{snap_min}.joblib não encontrado — encoding omitido")

        # ── Lista de features ──
        med_path = models_dir / f"train_medians_min{snap_min}.joblib"
        train_medians = joblib.load(med_path) if med_path.exists() else {}

        feature_list: list[str] = []
        if meta and snap_min in meta.get("models", {}):
            fl_candidate = meta["models"][snap_min].get("features") or []
            feature_list = [c for c in fl_candidate if c in df_te.columns]
        if not feature_list:
            # Fallback: infere features numéricas
            skip = ("target_", "snap_minute", "event_id", "kickoff")
            num  = df_tr.select_dtypes(include=[np.number]).columns.tolist()
            feature_list = [c for c in num
                            if not any(c.startswith(p) for p in skip) and c != TARGET]
            null_pcts    = df_tr[feature_list].isnull().mean()
            feature_list = [c for c in feature_list if null_pcts[c] < 0.99]
        print(f"  Features: {len(feature_list)}")

        global_mean = float(df_tr[TARGET].mean()) if TARGET in df_tr.columns else 11.0

        X_cal = _prepare_X(df_ca, feature_list, train_medians, global_mean)
        X_te  = _prepare_X(df_te, feature_list, train_medians, global_mean)

        y_te  = df_te[TARGET].values.astype(float)
        y_cal = df_ca[TARGET].values.astype(float)

        print(f"  Amostras: cal={len(X_cal)}  test={len(X_te)}")

        # ── Distribução NGBoost no test set ──
        print("  Calculando pred_dist no test set...")
        try:
            dist_test = ngb_model.pred_dist(X_te)
            dist_cal  = ngb_model.pred_dist(X_cal)
        except Exception as e:
            print(f"  [ERRO] pred_dist falhou: {e}. Pulando.")
            continue

        try:
            params_te  = dist_test.params
            params_cal = dist_cal.params
            s_te,  sc_te  = lognorm_params(params_te)
            s_cal, sc_cal = lognorm_params(params_cal)
        except Exception as e:
            print(f"  [ERRO] Extração de params falhou: {e}. Pulando.")
            continue

        mu_te  = lognorm_mean(s_te,  sc_te)
        mu_cal = lognorm_mean(s_cal, sc_cal)

        print(f"  E[Y] test: mean={mu_te.mean():.3f}  std={mu_te.std():.3f}")

        # ── PARTE A: PIT ──
        print("  [Parte A] Calculando PIT...")
        pit_vals   = compute_pit(y_te, s_te, sc_te, randomized=True)
        pit_result = analyse_pit(pit_vals, snap_min)
        print(f"    KS stat={pit_result['ks_stat']:.4f}  p={pit_result['ks_pval']:.6f}  "
              f"{'✅ uniforme' if pit_result['calibrated'] else '❌ NÃO uniforme'}")
        print(f"    Shape: {pit_result['shape']}")
        print(f"    Bias:  {pit_result['bias']}")

        # ── PARTE B: Reliability diagram ──
        print("  [Parte B] Calculando reliability...")
        lines_te = _dynamic_line(df_te, snap_min)
        lines_te = lines_te[:len(y_te)]
        actual_over = (y_te > lines_te).astype(float)

        # P(over) via CDF LogNorm nativa
        p_lognorm = p_over_lognorm(lines_te, s_te, sc_te)
        p_lognorm = np.clip(p_lognorm, 1e-6, 1 - 1e-6)

        # P(over) via Poisson(mu) — como o pipeline atual
        p_poisson = p_over_poisson_approx(lines_te, mu_te)
        p_poisson = np.clip(p_poisson, 1e-6, 1 - 1e-6)

        rel_lognorm = compute_reliability(p_lognorm,  actual_over)
        rel_poisson = compute_reliability(p_poisson,  actual_over)

        print(f"    ECE LogNorm  = {rel_lognorm['ece']:.5f}  "
              f"overconf={rel_lognorm['overconf']:+.5f}")
        print(f"    ECE Poisson  = {rel_poisson['ece']:.5f}  "
              f"overconf={rel_poisson['overconf']:+.5f}")

        # ── PARTE C: Métricas numéricas ──
        bias_mean  = float(np.mean(mu_te - y_te))
        overdisper = _overdispersion_ratio(y_te, mu_te)

        row = {
            **pit_result,
            "ece_lognorm":   rel_lognorm["ece"],
            "ece_poisson":   rel_poisson["ece"],
            "overconf_lognorm": rel_lognorm["overconf"],
            "overconf_poisson": rel_poisson["overconf"],
            "bias_mean":     round(bias_mean, 4),
            "overdispersion": round(overdisper, 4),
            "actual_over_rate": round(float(actual_over.mean()), 4),
            "pred_over_rate_lognorm": round(float((p_lognorm > 0.5).mean()), 4),
            "pred_over_rate_poisson": round(float((p_poisson > 0.5).mean()), 4),
        }
        # Remove colunas não-serializáveis para o dataframe
        row_csv = {k: v for k, v in row.items() if not isinstance(v, list)}
        diag_rows.append(row_csv)

        # ── Plots ──
        if not args.no_plots:
            print("  Gerando plots...")
            _make_plots(pit_result, rel_lognorm, rel_poisson, snap_min, plots_dir)

        conclusions.append(_interpret(row))

    # ── Salva CSV ──
    if diag_rows:
        report_df = pd.DataFrame(diag_rows)
        out_csv   = models_dir / "ngboost_calibration_report.csv"
        report_df.to_csv(out_csv, index=False)
        print(f"\n  Relatório salvo: {out_csv}")
        print_diagnostics_table(diag_rows)

    # ── Conclusões ──
    print("\n" + "=" * 90)
    print("  CONCLUSÕES E RECOMENDAÇÕES")
    print("=" * 90)
    for c in conclusions:
        print(c)

    print("""
─────────────────────────────────────────────────────────────────────────────
SNIPPET: Substituir Poisson approx por CDF LogNormal nativa no pipeline
(betsapi_corners_analysis.py, linhas ~2730-2733)

  # ANTES:
  p_ngb_test = np.array([1.0 - sp_poisson.cdf(fl, mu=max(m, 0.01))
                          for fl, m in zip(fl_test, _ngb_mu_test)])

  # DEPOIS (usa CDF nativa da LogNormal):
  _ngb_s_te    = _ngb_dist_test.params["s"]
  _ngb_sc_te   = _ngb_dist_test.params["scale"]
  p_ngb_test   = np.array([
      1.0 - sp_lognorm.cdf(max(float(fl) + 0.5, 0.01),
                           s=float(_ngb_s_te[i]),
                           scale=float(_ngb_sc_te[i]))
      for i, fl in enumerate(fl_test)
  ])
─────────────────────────────────────────────────────────────────────────────

Execute recalibrate_ngboost.py se ECE > 0.06 ou KS p < 0.05:
  python recalibrate_ngboost.py --walk-forward
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de calibração NGBoost")
    parser.add_argument("--parquet",     default=str(DEFAULT_PARQUET))
    parser.add_argument("--models-dir",  default=str(DEFAULT_MODELS))
    parser.add_argument("--plots-dir",   default=str(DEFAULT_PLOTS))
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--minutes",     type=int, nargs="+", default=None,
                        help="Minutos a analisar (padrão: 15 30 45 60 75)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
