"""
audit_team_encoding.py
======================
Auditoria e experimento A/B/C do target encoding de times/ligas.

Uso:
    # Parte 1 — auditoria do encoding atual
    python audit_team_encoding.py

    # Parte 2 — experimento A/B/C walk-forward (mais lento, ~10-20 min)
    python audit_team_encoding.py --experiment

    # Ambos juntos
    python audit_team_encoding.py --experiment

    # Opções
    python audit_team_encoding.py --parquet dados_escanteios/features_ml.parquet
    python audit_team_encoding.py --experiment --n-folds 5 --n-estimators 200
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

# Importa utilitários de ROI compartilhados (mesma lógica do pipeline principal)
try:
    from _roi_utils import (select_thresh, dline_vec as _dline_shared,
                            ODDS_OVER, BREAKEVEN, MIN_EDGE)
    ODDS = ODDS_OVER
    _HAS_ROI_UTILS = True
except ImportError:
    _HAS_ROI_UTILS = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET   = Path("dados_escanteios/features_ml.parquet")
DEFAULT_AUDIT_CSV = Path("dados_escanteios/team_encoding_audit.csv")
DEFAULT_EXP_CSV   = Path("dados_escanteios/ab_experiment_results.csv")
SNAPSHOT_MINUTES  = [15, 30, 45, 60, 75]
TARGET            = "target_corners_total"
ENCODE_COLS       = ["home_team", "away_team", "league_id"]
ENC_SMOOTHING     = 10    # igual ao pipeline principal
KF_SMOOTHING      = 20    # smoothing mais forte para k-fold TE
N_WF_FOLDS        = 7     # 7 folds → 5 janelas de teste (ti ∈ 2..6)
ODDS              = 1.83  # odds fixas para ROI simplificado
EDGE_THRESH       = 0.5   # mínimo |pred - linha| para apostar
XGB_DEFAULTS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)


# ---------------------------------------------------------------------------
# TARGET ENCODER — versão standalone (não depende do script principal)
# ---------------------------------------------------------------------------
class _TargetEncoderSmoothed:
    """Smoothed target encoder (fit no treino, transform em qualquer split)."""

    def __init__(self, cols: list[str], target_col: str, smoothing: int = 10):
        self.cols        = cols
        self.target_col  = target_col
        self.smoothing   = smoothing
        self.encodings_: dict[str, dict] = {}
        self.counts_:    dict[str, dict] = {}
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "_TargetEncoderSmoothed":
        self.global_mean_ = float(df[self.target_col].mean())
        for col in self.cols:
            if col not in df.columns:
                continue
            stats = df.groupby(col)[self.target_col].agg(["mean", "count"])
            smooth = stats["count"] / (stats["count"] + self.smoothing)
            stats["encoded"] = smooth * stats["mean"] + (1 - smooth) * self.global_mean_
            self.encodings_[col] = stats["encoded"].to_dict()
            self.counts_[col]    = stats["count"].to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols:
            mapping = self.encodings_.get(col, {})
            counts  = self.counts_.get(col, {})
            df[f"{col}_target_enc"] = df[col].map(mapping).fillna(self.global_mean_)
            raw_cnt = df[col].map(counts).fillna(0.0)
            df[f"{col}_enc_reliability"] = (raw_cnt / (raw_cnt + self.smoothing)).round(4)
        return df


def _kfold_target_encode(
    df_train: pd.DataFrame,
    cols: list[str],
    target_col: str,
    n_folds: int = 5,
    smoothing: int = 20,
) -> pd.DataFrame:
    """
    Out-of-fold target encoding no conjunto de treino.
    Para cal/test: usar _TargetEncoderSmoothed.fit(df_train).transform().
    """
    df_out      = df_train.copy()
    global_mean = float(df_train[target_col].mean())
    fold_idx    = np.arange(len(df_train)) % n_folds

    for col in cols:
        if col not in df_train.columns:
            continue
        enc_col = f"{col}_target_enc"
        rel_col = f"{col}_enc_reliability"
        df_out[enc_col] = global_mean
        df_out[rel_col] = 0.0

        for fold in range(n_folds):
            mask_val = fold_idx == fold
            mask_tr  = ~mask_val
            if mask_tr.sum() == 0:
                continue
            stats = (
                df_train.loc[mask_tr]
                .groupby(col)[target_col]
                .agg(["mean", "count"])
            )
            smooth  = stats["count"] / (stats["count"] + smoothing)
            enc_map = (smooth * stats["mean"] + (1 - smooth) * global_mean).to_dict()
            cnt_map = stats["count"].to_dict()

            mapped = df_train.loc[mask_val, col].map(enc_map).fillna(global_mean)
            df_out.loc[mask_val, enc_col] = mapped.values

            raw_cnt = df_train.loc[mask_val, col].map(cnt_map).fillna(0.0)
            df_out.loc[mask_val, rel_col] = (raw_cnt / (raw_cnt + smoothing)).values

    return df_out


# ---------------------------------------------------------------------------
# FEATURE PREP — versão mínima (sem depender do script principal)
# ---------------------------------------------------------------------------
def _prepare_features(
    df: pd.DataFrame,
    target: str,
    available_override: list[str] | None = None,
    medians: dict | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Filtra features numéricas, imputa NaN, retorna (feature_list, df_limpo)."""
    SKIP_PREFIXES = ("target_", "snap_minute", "event_id", "kickoff")

    if available_override is not None:
        available = [c for c in available_override if c in df.columns]
    else:
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        available = [
            c for c in num_cols
            if not any(c.startswith(p) for p in SKIP_PREFIXES) and c != target
        ]
        # Remove colunas com >99% NaN
        null_pcts = df[available].isnull().mean()
        available = [c for c in available if null_pcts[c] < 0.99]

    df_out = df[available + [target]].copy()
    df_out = df_out.dropna(subset=[target])

    # Imputa
    fill_med = [c for c in available if c.startswith(("hist_", "league_", "h2h_"))
                or c.endswith("_target_enc")]
    fill_zero = [c for c in available if c not in fill_med]

    for c in fill_med:
        val = medians[c] if (medians and c in medians) else df_out[c].median()
        df_out[c] = df_out[c].fillna(val)
    for c in fill_zero:
        df_out[c] = df_out[c].fillna(0)

    df_out[available] = df_out[available].replace([np.inf, -np.inf], np.nan).fillna(0)
    return available, df_out


# ---------------------------------------------------------------------------
# PART 1 — AUDITORIA DO ENCODING
# ---------------------------------------------------------------------------
def _split_temporal(df_min: pd.DataFrame):
    """Mesmo split temporal do pipeline principal (60/20/20)."""
    n     = len(df_min)
    n_te  = int(n * 0.20)
    n_cal = int((n - n_te) * 0.25)
    n_tr  = n - n_te - n_cal
    df_tr = df_min.iloc[:n_tr].copy()
    df_ca = df_min.iloc[n_tr:n_tr + n_cal].copy()
    df_te = df_min.iloc[n_tr + n_cal:].copy()
    return df_tr, df_ca, df_te


def run_audit(df: pd.DataFrame) -> pd.DataFrame:
    enc_cols_present = [c for c in ENCODE_COLS if c in df.columns]
    if not enc_cols_present:
        print("[AVISO] Nenhuma das colunas de encoding encontrada no parquet:",
              ENCODE_COLS)
        return pd.DataFrame()

    print(f"\nColunas de encoding encontradas: {enc_cols_present}")
    rows: list[dict] = []

    for snap_min in SNAPSHOT_MINUTES:
        print(f"\n{'─'*60}")
        print(f"  Minuto {snap_min}")
        print(f"{'─'*60}")

        df_min = df[df["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)

        df_tr, df_ca, df_te = _split_temporal(df_min)

        te = _TargetEncoderSmoothed(enc_cols_present, TARGET, ENC_SMOOTHING)
        te.fit(df_tr)

        df_tr_enc = te.transform(df_tr)
        df_ca_enc = te.transform(df_ca)
        df_te_enc = te.transform(df_te)

        for col in enc_cols_present:
            enc_col = f"{col}_target_enc"
            rel_col = f"{col}_enc_reliability"

            # ── Estatísticas de distribuição do encoding ──
            enc_vals = df_tr_enc[enc_col]
            stats_enc = {
                "snap_min": snap_min,
                "column":   col,
                "n_treino": len(df_tr),
                "n_cal":    len(df_ca),
                "n_teste":  len(df_te),
                "enc_mean": round(float(enc_vals.mean()), 4),
                "enc_std":  round(float(enc_vals.std()),  4),
                "enc_min":  round(float(enc_vals.min()),  4),
                "enc_p25":  round(float(enc_vals.quantile(0.25)), 4),
                "enc_p50":  round(float(enc_vals.quantile(0.50)), 4),
                "enc_p75":  round(float(enc_vals.quantile(0.75)), 4),
                "enc_max":  round(float(enc_vals.max()),  4),
            }

            # ── Contagem de entidades únicas e cauda longa ──
            unique_total  = int(df_tr[col].nunique())
            counts_series = df_tr.groupby(col)[TARGET].count()
            rare_lt10     = int((counts_series < 10).sum())
            rare_lt30     = int((counts_series < 30).sum())

            stats_enc["unique_in_train"] = unique_total
            stats_enc["rare_lt10"]       = rare_lt10
            stats_enc["rare_lt30"]       = rare_lt30
            stats_enc["pct_rare_lt10"]   = round(rare_lt10 / max(unique_total, 1), 4)

            # ── Correlação encoding ↔ target (treino / cal / teste) ──
            def _corr(df_enc, split_name):
                s = df_enc[enc_col].corr(df_enc[TARGET])
                return round(float(s) if not np.isnan(s) else 0.0, 4)

            stats_enc["corr_train"] = _corr(df_tr_enc, "treino")
            stats_enc["corr_cal"]   = _corr(df_ca_enc, "cal")
            stats_enc["corr_test"]  = _corr(df_te_enc, "teste")
            stats_enc["corr_drift"] = round(
                abs(stats_enc["corr_train"] - stats_enc["corr_test"]), 4)

            # ── Spearman por bucket de frequência ──
            bucket_results = {}
            for bname, blo, bhi in [
                ("<10",   0, 9), ("10-30", 10, 30),
                ("30-100", 31, 100), (">100", 101, 999_999)
            ]:
                ents = counts_series[(counts_series >= blo) & (counts_series <= bhi)].index
                mask = df_te_enc[col].isin(ents)
                sub  = df_te_enc[mask]
                if len(sub) >= 10:
                    r, pval = sp_stats.spearmanr(sub[enc_col], sub[TARGET])
                    bucket_results[f"spearman_{bname}"] = round(float(r), 4)
                    bucket_results[f"n_{bname}"]        = len(sub)
                else:
                    bucket_results[f"spearman_{bname}"] = float("nan")
                    bucket_results[f"n_{bname}"]        = int(mask.sum())
            stats_enc.update(bucket_results)

            rows.append(stats_enc)

            # ── Print resumo ──
            print(f"\n  [{col}]")
            print(f"    Treino: {unique_total} únicos  |  <10 jogos: {rare_lt10} ({stats_enc['pct_rare_lt10']:.1%})")
            print(f"    Correlação enc↔target: treino={stats_enc['corr_train']:.4f}  "
                  f"cal={stats_enc['corr_cal']:.4f}  teste={stats_enc['corr_test']:.4f}  "
                  f"drift={stats_enc['corr_drift']:.4f}")
            print(f"    Spearman por freq: <10={bucket_results.get('spearman_<10','?'):.4f}  "
                  f"10-30={bucket_results.get('spearman_10-30','?'):.4f}  "
                  f"30-100={bucket_results.get('spearman_30-100','?'):.4f}  "
                  f">100={bucket_results.get('spearman_>100','?'):.4f}")

            # ── Top-20 times com maior encoding (apenas home_team e away_team) ──
            if col in ("home_team", "away_team") and enc_col in df_tr_enc.columns:
                _print_top20_audit(df_tr_enc, df_te_enc, col, enc_col, te)

    report = pd.DataFrame(rows)
    return report


def _print_top20_audit(
    df_tr_enc: pd.DataFrame,
    df_te_enc: pd.DataFrame,
    col: str,
    enc_col: str,
    te: _TargetEncoderSmoothed,
) -> None:
    """Imprime top-20 times por encoding vs realidade out-of-sample."""
    # Reconstrói mapeamento enc_val por entidade
    enc_map   = te.encodings_.get(col, {})
    cnt_map   = te.counts_.get(col, {})
    global_m  = te.global_mean_

    if not enc_map:
        return

    enc_df = (
        pd.DataFrame({"entity": list(enc_map.keys()),
                      "enc_value": list(enc_map.values())})
        .assign(n_train=lambda x: x["entity"].map(cnt_map).fillna(0).astype(int))
        .sort_values("enc_value", ascending=False)
        .head(20)
    )

    # Média real no conjunto de TESTE para esses times
    if col in df_te_enc.columns and TARGET in df_te_enc.columns:
        real_test = (
            df_te_enc.groupby(col)[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "real_mean_test", "count": "n_test"})
        )
        enc_df = enc_df.merge(
            real_test, left_on="entity", right_index=True, how="left"
        )
        enc_df["abs_error"] = (enc_df["enc_value"] - enc_df["real_mean_test"]).abs().round(3)
    else:
        enc_df["real_mean_test"] = float("nan")
        enc_df["n_test"]         = 0
        enc_df["abs_error"]      = float("nan")

    print(f"\n    Top-20 por {enc_col} (vs realidade out-of-sample):")
    print(f"    {'Entity':<30} {'Enc':>7} {'N_train':>8} {'Real_OOS':>9} {'N_test':>7} {'|Err|':>7}")
    print(f"    {'─'*30} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*7}")
    for _, row in enc_df.iterrows():
        real = f"{row['real_mean_test']:.3f}" if not pd.isna(row.get("real_mean_test")) else "   N/A"
        err  = f"{row['abs_error']:.3f}"      if not pd.isna(row.get("abs_error"))      else "   N/A"
        n_te = int(row["n_test"]) if not pd.isna(row.get("n_test")) else 0
        print(f"    {str(row['entity']):<30} {row['enc_value']:>7.3f} "
              f"{int(row['n_train']):>8,} {real:>9} {n_te:>7,} {err:>7}")


# ---------------------------------------------------------------------------
# PART 2 — EXPERIMENTO A/B/C WALK-FORWARD
# ---------------------------------------------------------------------------
def _dynamic_line(
    df: pd.DataFrame,
    snap_min: int,
    fallback_rate: float = 11.0 / 90.0,
) -> np.ndarray:
    """Linha dinâmica: corners_atual + tempo_restante × taxa."""
    rem = 90 - snap_min
    csf = (df["corners_total_so_far"].values
           if "corners_total_so_far" in df.columns
           else np.zeros(len(df)))
    if "league_avg_corners" in df.columns:
        rate = np.where(
            df["league_avg_corners"].isna(),
            fallback_rate,
            df["league_avg_corners"].values / 90.0,
        )
    else:
        rate = np.full(len(df), fallback_rate)
    return csf + rem * rate


def _roi_from_preds(
    preds: np.ndarray,
    y_true: np.ndarray,
    lines: np.ndarray,
    edge: float = EDGE_THRESH,
    odds: float = ODDS,
) -> tuple[float, int]:
    """ROI simplificado: aposta quando |pred - linha| > edge."""
    diffs  = preds - lines
    bets   = np.abs(diffs) > edge
    n_bets = int(bets.sum())
    if n_bets == 0:
        return float("nan"), 0

    over_bet  = bets & (diffs > 0)
    under_bet = bets & (diffs < 0)
    actual_over = y_true > lines

    wins = (over_bet & actual_over) | (under_bet & ~actual_over)
    profit = (wins[bets].astype(float) * (odds - 1)
              - (~wins[bets]).astype(float)).sum()
    return float(profit / n_bets), n_bets


def _run_wf_method(
    df_min: pd.DataFrame,
    snap_min: int,
    method: str,          # "A", "B", "C"
    enc_cols: list[str],
    n_wf_folds: int,
    n_estimators: int,
    verbose: bool = True,
) -> dict:
    """Walk-forward para um método e um snap_minute."""
    try:
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        print(f"  [ERRO] {e}")
        return {}

    fold_size = len(df_min) // n_wf_folds
    fold_maes: list[float] = []
    fold_rois: list[float] = []
    fold_bets: list[int]   = []

    hp = {**XGB_DEFAULTS, "n_estimators": n_estimators, "early_stopping_rounds": 30}

    for ti in range(2, n_wf_folds):
        cal_start  = (ti - 1) * fold_size
        cal_end    = ti * fold_size
        test_start = ti * fold_size
        test_end   = (ti + 1) * fold_size if ti < n_wf_folds - 1 else len(df_min)

        df_tr = df_min.iloc[:cal_start].copy()
        df_ca = df_min.iloc[cal_start:cal_end].copy()
        df_te = df_min.iloc[test_start:test_end].copy()

        if len(df_tr) < 80 or len(df_ca) < 20 or len(df_te) < 20:
            if verbose:
                print(f"      fold {ti}: dados insuficientes, pulando.")
            continue

        # ── Encoding por método ──
        if method == "A":
            # Baseline: TE smoothed fit no treino
            te_a = _TargetEncoderSmoothed(enc_cols, TARGET, ENC_SMOOTHING)
            te_a.fit(df_tr)
            df_tr = te_a.transform(df_tr)
            df_ca = te_a.transform(df_ca)
            df_te = te_a.transform(df_te)

        elif method == "B":
            # Sem encoding: remove colunas enc se existirem
            enc_drop = []
            for c in enc_cols:
                enc_drop += [f"{c}_target_enc", f"{c}_enc_reliability"]
            df_tr = df_tr.drop(columns=[c for c in enc_drop if c in df_tr.columns])
            df_ca = df_ca.drop(columns=[c for c in enc_drop if c in df_ca.columns])
            df_te = df_te.drop(columns=[c for c in enc_drop if c in df_te.columns])

        elif method == "C":
            # K-fold TE (5 folds) dentro do treino
            df_tr = _kfold_target_encode(df_tr, enc_cols, TARGET, n_folds=5, smoothing=KF_SMOOTHING)
            # Cal e test: TE fit no treino completo
            te_c = _TargetEncoderSmoothed(enc_cols, TARGET, KF_SMOOTHING)
            te_c.fit(df_tr)   # fit nos dados já com enc (enc_col existe, mas valores são OOF)
            # Refaz o fit nos dados raw antes de adicionar as colunas OOF para evitar leakage
            # Usamos df_min.iloc[:cal_start] sem transform
            df_tr_raw = df_min.iloc[:cal_start].copy()
            te_c_raw = _TargetEncoderSmoothed(enc_cols, TARGET, KF_SMOOTHING)
            te_c_raw.fit(df_tr_raw)
            df_ca = te_c_raw.transform(df_ca)
            df_te = te_c_raw.transform(df_te)
            # df_tr já tem OOF encoding

        # ── Feature prep ──
        avail_tr, df_tr_c = _prepare_features(df_tr, TARGET)
        med_wf = {c: df_tr_c[c].median()
                  for c in avail_tr
                  if c.startswith(("hist_", "league_", "h2h_")) or c.endswith("_target_enc")}
        _, df_ca_c = _prepare_features(df_ca, TARGET, avail_tr, med_wf)
        _, df_te_c = _prepare_features(df_te, TARGET, avail_tr, med_wf)

        if len(df_te_c) < 20:
            continue

        X_tr, y_tr = df_tr_c[avail_tr], df_tr_c[TARGET]
        X_ca, y_ca = df_ca_c[avail_tr], df_ca_c[TARGET]
        X_te, y_te = df_te_c[avail_tr], df_te_c[TARGET]

        model = XGBRegressor(**hp)
        model.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)

        raw_te = model.predict(X_te)
        raw_ca = model.predict(X_ca)
        mae_raw = float(mean_absolute_error(y_te, raw_te))

        # Calibração isotônica
        iso = IsotonicRegression(y_min=0, y_max=35, out_of_bounds="clip")
        iso.fit(raw_ca, y_ca)
        cal_te  = iso.predict(raw_te)
        mae_cal = float(mean_absolute_error(y_te, cal_te))
        preds   = cal_te if mae_cal < mae_raw else raw_te
        mae_f   = min(mae_raw, mae_cal)

        # ROI
        # Reconstrói df_te sem o TARGET para obter a linha dinâmica do df original
        orig_te_idx = df_min.index[test_start:test_end]
        df_te_orig  = df_min.loc[orig_te_idx]
        lines  = _dynamic_line(df_te_orig, snap_min)
        # Alinha indices com df_te_c
        lines_aligned = lines[:len(y_te)]
        roi_f, n_bets_f = _roi_from_preds(preds, y_te.values, lines_aligned)

        fold_maes.append(mae_f)
        fold_rois.append(roi_f if not np.isnan(roi_f) else 0.0)
        fold_bets.append(n_bets_f)

        if verbose:
            roi_str = f"{roi_f:+.1%}" if not np.isnan(roi_f) else "  N/A"
            print(f"      fold {ti}/{n_wf_folds-1}: MAE={mae_f:.3f}  "
                  f"ROI={roi_str}  bets={n_bets_f}  feats={len(avail_tr)}")

    if not fold_maes:
        return {}

    roi_arr = np.array(fold_rois)
    return {
        "snap_min":   snap_min,
        "method":     method,
        "mae_mean":   round(float(np.mean(fold_maes)), 4),
        "mae_std":    round(float(np.std(fold_maes)),  4),
        "roi_mean":   round(float(np.mean(roi_arr)),   4),
        "roi_std":    round(float(np.std(roi_arr)),    4),
        "roi_ic90_lo": round(float(np.percentile(roi_arr, 5)),  4),
        "roi_ic90_hi": round(float(np.percentile(roi_arr, 95)), 4),
        "n_bets_total": int(np.sum(fold_bets)),
        "n_folds_ok":  len(fold_maes),
    }


def run_experiment(
    df: pd.DataFrame,
    n_estimators: int = 300,
    n_wf_folds: int   = N_WF_FOLDS,
) -> pd.DataFrame:
    try:
        from xgboost import XGBRegressor  # noqa — just check import
    except ImportError:
        print("[ERRO] xgboost não disponível. Instale: pip install xgboost")
        return pd.DataFrame()

    enc_cols_present = [c for c in ENCODE_COLS if c in df.columns]
    methods = ["A", "B", "C"]
    method_labels = {
        "A": "BASELINE (TE atual)",
        "B": "SEM encoding",
        "C": "K-Fold TE (5 folds, smooth=20)",
    }

    all_results: list[dict] = []

    for snap_min in SNAPSHOT_MINUTES:
        df_min = df[df["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)

        if len(df_min) < 500:
            print(f"  [min{snap_min}] Dados insuficientes ({len(df_min)}), pulando.")
            continue

        for method in methods:
            label = method_labels[method]
            print(f"\n{'═'*70}")
            print(f"  Minuto {snap_min}  |  Método {method}: {label}")
            print(f"  n_estimators={n_estimators}  folds={n_wf_folds}")
            print(f"{'─'*70}")

            result = _run_wf_method(
                df_min        = df_min,
                snap_min      = snap_min,
                method        = method,
                enc_cols      = enc_cols_present,
                n_wf_folds    = n_wf_folds,
                n_estimators  = n_estimators,
                verbose       = True,
            )
            if result:
                all_results.append(result)

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# PART 3 — COMPARAÇÃO E DECISÃO
# ---------------------------------------------------------------------------
def print_comparison_table(exp_df: pd.DataFrame) -> None:
    if exp_df.empty:
        return

    print("\n" + "=" * 90)
    print("  TABELA COMPARATIVA A/B/C  (Δ = diferença vs Método A / BASELINE)")
    print("=" * 90)

    for snap_min in SNAPSHOT_MINUTES:
        sub = exp_df[exp_df["snap_min"] == snap_min]
        if sub.empty:
            continue

        row_a = sub[sub["method"] == "A"]
        if row_a.empty:
            continue

        mae_a = float(row_a["mae_mean"].iloc[0])
        roi_a = float(row_a["roi_mean"].iloc[0])

        print(f"\n  Minuto {snap_min}")
        print(f"  {'Método':<35} {'MAE':>7} {'Δ MAE':>8} {'ROI':>8} {'Δ ROI':>8} "
              f"{'ROI_std':>8} {'IC90_lo':>8} {'IC90_hi':>8} {'Bets':>7}")
        print(f"  {'─'*35} {'─'*7} {'─'*8} {'─'*8} {'─'*8} "
              f"{'─'*8} {'─'*8} {'─'*8} {'─'*7}")

        for _, row in sub.iterrows():
            mae = float(row["mae_mean"])
            roi = float(row["roi_mean"])
            d_mae = mae - mae_a
            d_roi = roi - roi_a
            method_lbl = {
                "A": "A: BASELINE (TE atual)",
                "B": "B: SEM encoding",
                "C": "C: K-Fold TE",
            }.get(row["method"], row["method"])

            print(f"  {method_lbl:<35} {mae:>7.4f} {d_mae:>+8.4f} {roi:>+7.1%} "
                  f"{d_roi:>+7.1%} {float(row['roi_std']):>+7.1%} "
                  f"{float(row['roi_ic90_lo']):>+7.1%} "
                  f"{float(row['roi_ic90_hi']):>+7.1%} "
                  f"{int(row['n_bets_total']):>7,}")

    print()


def print_decision(exp_df: pd.DataFrame) -> None:
    if exp_df.empty:
        return

    print("\n" + "=" * 90)
    print("  DECISÃO FINAL")
    print("=" * 90)

    decisions: list[str] = []

    for snap_min in SNAPSHOT_MINUTES:
        sub   = exp_df[exp_df["snap_min"] == snap_min]
        row_a = sub[sub["method"] == "A"]
        row_b = sub[sub["method"] == "B"]
        row_c = sub[sub["method"] == "C"]

        if row_a.empty:
            continue

        mae_a  = float(row_a["mae_mean"].iloc[0])
        roi_a  = float(row_a["roi_mean"].iloc[0])
        std_a  = float(row_a["roi_std"].iloc[0])

        print(f"\n  Minuto {snap_min}:")

        if not row_b.empty:
            mae_b  = float(row_b["mae_mean"].iloc[0])
            roi_b  = float(row_b["roi_mean"].iloc[0])
            d_mae  = mae_b - mae_a
            d_roi  = roi_b - roi_a
            if d_mae <= 0.03 * mae_a and abs(d_roi) <= 0.02:
                print(f"    ✅ DECISÃO B: REMOVER encoding. "
                      f"Δ MAE={d_mae:+.4f} ({d_mae/mae_a:+.1%})  Δ ROI={d_roi:+.2%}")
                decisions.append(f"min{snap_min}: REMOVER encoding (B)")
            elif not row_c.empty:
                roi_c = float(row_c["roi_mean"].iloc[0])
                std_c = float(row_c["roi_std"].iloc[0])
                d_roi_c = roi_c - roi_a
                if d_roi_c > 0 and std_c < std_a:
                    print(f"    ✅ DECISÃO C: K-FOLD TE. "
                          f"Δ ROI={d_roi_c:+.2%}  std: {std_a:.3f}→{std_c:.3f} ↓")
                    decisions.append(f"min{snap_min}: K-FOLD TE (C)")
                else:
                    print(f"    ℹ  DECISÃO A: Manter TE atual. "
                          f"B piora MAE {d_mae:+.4f}  C: Δ ROI={d_roi_c:+.2%} std={std_c:.3f}")
                    decisions.append(f"min{snap_min}: MANTER TE atual (A)")
            else:
                print(f"    ℹ  DECISÃO A: Manter TE atual (sem dados de C).")
                decisions.append(f"min{snap_min}: MANTER TE atual (A)")
        else:
            print("    ⚠ Sem dados de B.")

    print("\n  RESUMO:")
    for d in decisions:
        print(f"    • {d}")

    # ── Snippets ──
    print("\n" + "=" * 90)
    print("  SNIPPETS PRONTOS PARA SUBSTITUIÇÃO EM betsapi_corners_analysis.py")
    print("=" * 90)

    print("""
── SNIPPET B: Remover encoding completamente ─────────────────────────────────
  Localizar ≈ linha 2190 (loop for snap_min in SNAPSHOT_MINUTES):

  # ANTES:
  encode_cols_avail = [c for c in ENCODE_COLS if c in df_min.columns]
  if encode_cols_avail:
      te_min = TargetEncoderSmoothed(...)
      te_min.fit(df_train_raw)
      df_train_raw = te_min.transform(df_train_raw)
      df_cal_raw   = te_min.transform(df_cal_raw)
      df_test_raw  = te_min.transform(df_test_raw)
  else:
      te_min = None

  # DEPOIS (sem encoding):
  te_min = None   # encoding removido — ver audit_team_encoding.py
""")

    print("""
── SNIPPET C: K-Fold TE (5 folds, smoothing=20) ──────────────────────────────
  Adicionar a função _kfold_target_encode() deste arquivo ao script principal.
  Localizar ≈ linha 2190:

  # ANTES:
  if encode_cols_avail:
      te_min = TargetEncoderSmoothed(cols=encode_cols_avail, target_col=TARGET,
                                     smoothing=10)
      te_min.fit(df_train_raw)
      df_train_raw = te_min.transform(df_train_raw)
      ...

  # DEPOIS (K-Fold TE):
  if encode_cols_avail:
      # Out-of-fold encoding no treino (evita leakage intrafold)
      df_train_raw = _kfold_target_encode(
          df_train_raw, encode_cols_avail, TARGET, n_folds=5, smoothing=20
      )
      # Cal e test: fit no treino completo + transform
      te_min = TargetEncoderSmoothed(cols=encode_cols_avail, target_col=TARGET,
                                     smoothing=20)
      te_min.fit(df_train_raw)
      df_cal_raw  = te_min.transform(df_cal_raw)
      df_test_raw = te_min.transform(df_test_raw)
  else:
      te_min = None
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Auditoria de target encoding")
    parser.add_argument("--parquet",      default=str(DEFAULT_PARQUET))
    parser.add_argument("--audit-csv",    default=str(DEFAULT_AUDIT_CSV))
    parser.add_argument("--exp-csv",      default=str(DEFAULT_EXP_CSV))
    parser.add_argument("--experiment",   action="store_true",
                        help="Executa experimento A/B/C walk-forward")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Estimators para o experimento (padrão: 300)")
    parser.add_argument("--n-folds",      type=int, default=N_WF_FOLDS,
                        help=f"Folds walk-forward (padrão: {N_WF_FOLDS})")
    parser.add_argument("--skip-audit",   action="store_true",
                        help="Pula a Parte 1 (auditoria) e vai direto ao experimento")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}")
        sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df):,} linhas × {df.shape[1]} colunas")

    # Verifica colunas de encoding
    enc_present = [c for c in ENCODE_COLS if c in df.columns]
    if not enc_present:
        print(f"[AVISO] Nenhuma coluna de encoding ({ENCODE_COLS}) encontrada no parquet.")
        print("  O target encoding é aplicado durante o treino (betsapi_corners_analysis.py)")
        print("  e não está salvo no parquet. A auditoria usará os valores raw dos times.\n")

    if "snap_minute" in df.columns:
        dist = df["snap_minute"].value_counts().sort_index()
        print(f"snap_minute: {dict(dist)}\n")

    # ── Parte 1: Auditoria ──
    if not args.skip_audit:
        print("\n" + "═" * 70)
        print("  PARTE 1 — AUDITORIA DO ENCODING")
        print("═" * 70)
        report = run_audit(df)
        if not report.empty:
            audit_path = Path(args.audit_csv)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(audit_path, index=False)
            print(f"\n  Auditoria salva em: {audit_path}")

            # Print tabela resumo
            print("\n" + "─" * 90)
            print("  RESUMO: Correlação encoding ↔ target por minuto")
            print("─" * 90)
            pivot_cols = ["snap_min", "column", "corr_train", "corr_cal",
                          "corr_test", "corr_drift", "rare_lt10", "pct_rare_lt10",
                          "spearman_<10", "spearman_>100"]
            avail_pivot = [c for c in pivot_cols if c in report.columns]
            with pd.option_context("display.max_rows", 100, "display.width", 160,
                                   "display.float_format", "{:.4f}".format):
                print(report[avail_pivot].to_string(index=False))
        else:
            print("  Nenhuma coluna de encoding encontrada para auditar.")

    # ── Parte 2 + 3: Experimento A/B/C ──
    if args.experiment:
        print("\n" + "═" * 70)
        print("  PARTE 2 — EXPERIMENTO A/B/C  (walk-forward)")
        print(f"  n_estimators={args.n_estimators}  n_folds={args.n_folds}")
        print("═" * 70)

        exp_df = run_experiment(df, args.n_estimators, args.n_folds)

        if not exp_df.empty:
            exp_path = Path(args.exp_csv)
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            exp_df.to_csv(exp_path, index=False)
            print(f"\n  Resultados salvos em: {exp_path}")

            print_comparison_table(exp_df)
            print_decision(exp_df)
        else:
            print("  Experimento não produziu resultados.")
    else:
        print("\n  (Para rodar o experimento A/B/C: adicione a flag --experiment)\n")


if __name__ == "__main__":
    main()
