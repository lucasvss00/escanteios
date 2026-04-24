"""
validate_min45.py
=================
Validação profunda do modelo do minuto 45.

Uso:
    python validate_min45.py                           # análise completa
    python validate_min45.py --no-plots                # sem matplotlib
    python validate_min45.py --skip-negbinom           # pula teste NegBinom
    python validate_min45.py --minutes 30 45 60        # compara vizinhos
    python validate_min45.py --parquet dados_escanteios/features_ml.parquet
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import poisson as sp_poisson
from scipy.stats import nbinom as sp_nbinom

warnings.filterwarnings("ignore")

# Importa utilitários de ROI compartilhados (mesma lógica do pipeline principal)
try:
    from _roi_utils import (select_thresh, profit_vec,
                            ODDS_OVER, ODDS_UNDER, BREAKEVEN, MIN_EDGE)
    ODDS = ODDS_OVER
    _HAS_ROI_UTILS = True
except ImportError:
    _HAS_ROI_UTILS = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET  = Path("dados_escanteios/features_ml.parquet")
DEFAULT_MODELS   = Path("dados_escanteios")
DEFAULT_PLOTS    = Path("dados_escanteios/plots/min45_validation")
SNAPSHOT_MINUTES = [15, 30, 45, 60, 75]
TARGET           = "target_corners_total"
ODDS             = 1.83
BREAKEVEN        = 1.0 / ODDS      # ≈ 0.546
MIN_EDGE         = 0.02
N_WF_FOLDS       = 7               # 5 janelas de teste (ti ∈ 2..6)
BOOTSTRAP_N      = 1000
BOOTSTRAP_SEED   = 42


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _split_temporal(df_min: pd.DataFrame):
    n    = len(df_min)
    n_te = int(n * 0.20)
    n_ca = int((n - n_te) * 0.25)
    n_tr = n - n_te - n_ca
    return (
        df_min.iloc[:n_tr].copy(),
        df_min.iloc[n_tr:n_tr + n_ca].copy(),
        df_min.iloc[n_tr + n_ca:].copy(),
    )


def _prepare_X(df, feature_list, medians, global_mean):
    df2 = df.copy()
    for c in feature_list:
        if c not in df2.columns:
            df2[c] = 0.0
    fill_med  = [c for c in feature_list
                 if c.startswith(("hist_", "league_", "h2h_")) or c.endswith("_target_enc")]
    fill_zero = [c for c in feature_list if c not in fill_med]
    for c in fill_med:
        df2[c] = df2[c].fillna(medians.get(c, global_mean))
    for c in fill_zero:
        df2[c] = df2[c].fillna(0)
    df2[feature_list] = df2[feature_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df2[feature_list].values.astype(float)


def _dline_vec(csf_arr, la_arr_or_none, rem):
    lines = []
    la_arr = la_arr_or_none if la_arr_or_none is not None else [None] * len(csf_arr)
    for csf, la in zip(csf_arr, la_arr):
        try:
            rate = float(la) / 90.0 if (la is not None and np.isfinite(float(la))
                                         and float(la) > 0) else 0.1
        except (TypeError, ValueError):
            rate = 0.1
        lines.append(np.round((float(csf) + rem * rate) * 2) / 2)
    return np.array(lines, dtype=float)


def _get_dline(df, snap_min):
    rem = 90 - snap_min
    csf = (df["corners_total_so_far"].values
           if "corners_total_so_far" in df.columns else np.zeros(len(df)))
    la  = (df["league_avg_corners"].values.tolist()
           if "league_avg_corners" in df.columns else None)
    return _dline_vec(csf, la, rem)


def _p_over_poisson(line, mu):
    fl = np.floor(np.asarray(line)).astype(int)
    mu = np.asarray(mu)
    return np.array([1.0 - sp_poisson.cdf(int(fl[i]), mu=max(float(mu[i]), 0.01))
                     for i in range(len(mu))])


def _p_over_negbinom(line, mu, sigma):
    """NegBinom P(over) com r per-game derivado do sigma."""
    fl    = np.floor(np.asarray(line)).astype(int)
    mu    = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    resid_var = float(np.mean(sigma ** 2))
    mu_mean   = float(np.mean(mu))
    r_fallback = float(np.clip(mu_mean**2 / max(resid_var - mu_mean, 0.5), 0.5, 100))
    probs = np.empty(len(mu))
    for i in range(len(mu)):
        m  = max(float(mu[i]), 0.01)
        s2 = float(sigma[i]) ** 2
        r  = min(max(m**2 / (s2 - m), 0.5), 100.0) if s2 > m else r_fallback
        p  = r / (r + m)
        probs[i] = 1.0 - sp_nbinom.cdf(int(fl[i]), n=r, p=p)
    return probs


def _bootstrap_roi(n_bets, n_wins, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Bootstrap IC90% do ROI."""
    if n_bets < 5:
        roi = (n_wins * (ODDS - 1) - (n_bets - n_wins)) / max(n_bets, 1)
        return roi, roi, roi
    rng = np.random.default_rng(seed)
    outcomes = np.array([1.0] * n_wins + [0.0] * (n_bets - n_wins))
    samples  = []
    for _ in range(n_bootstrap):
        s   = rng.choice(outcomes, size=n_bets, replace=True)
        w_s = s.sum()
        samples.append((w_s * (ODDS - 1) - (n_bets - w_s)) / n_bets)
    arr = np.array(samples)
    roi_obs = (n_wins * (ODDS - 1) - (n_bets - n_wins)) / n_bets
    return (roi_obs,
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 95)))


def _eval_threshold(p_over_te, over_actual_te, thresh):
    """Métricas de apostas a um dado threshold."""
    mask   = p_over_te >= thresh
    n_bets = int(mask.sum())
    if n_bets == 0:
        return {"n_bets": 0, "acc": 0.0, "roi": 0.0, "ic90_lo": 0.0, "ic90_hi": 0.0,
                "zscore": 0.0}
    n_wins = int(over_actual_te[mask].sum())
    roi, ic_lo, ic_hi = _bootstrap_roi(n_bets, n_wins)
    acc  = n_wins / n_bets
    # Z-score vs ROI=0 (Ho: ROI <= 0)
    p_win_null = BREAKEVEN  # ROI=0 implica acurácia = breakeven
    se   = np.sqrt(p_win_null * (1 - p_win_null) / n_bets)
    z    = (acc - p_win_null) / max(se, 1e-9)
    return {"n_bets": n_bets, "acc": round(acc, 4), "roi": round(roi, 4),
            "ic90_lo": round(ic_lo, 4), "ic90_hi": round(ic_hi, 4), "zscore": round(z, 3)}


# ---------------------------------------------------------------------------
# PARTE 1 — SENSIBILIDADE AO THRESHOLD (walk-forward)
# ---------------------------------------------------------------------------
def _wf_single_fold(
    df_tr, df_ca, df_te,
    snap_min: int,
    te_model, medians, feature_list: list[str],
    global_mean: float,
):
    """
    Walk-forward de 1 fold.
    Retorna (p_over_te, over_actual_te, p_over_cal, over_actual_cal, len_test).
    """
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    if te_model is not None:
        # Recria TE no treino do fold (evita leakage)
        _cols = [c for c in te_model.cols if c in df_tr.columns]
        if _cols:
            _te = _TELocal(_cols, TARGET)
            _te.fit(df_tr)
            df_tr = _te.transform(df_tr)
            df_ca = _te.transform(df_ca)
            df_te = _te.transform(df_te)

    feats = [c for c in feature_list if c in df_te.columns]
    if not feats:
        return None, None, None, None, 0

    X_tr = _prepare_X(df_tr, feats, medians, global_mean)
    X_ca = _prepare_X(df_ca, feats, medians, global_mean)
    X_te = _prepare_X(df_te, feats, medians, global_mean)
    y_tr = df_tr[TARGET].values.astype(float)
    y_ca = df_ca[TARGET].values.astype(float)
    y_te = df_te[TARGET].values.astype(float)

    if len(X_tr) < 80 or len(X_ca) < 20 or len(X_te) < 20:
        return None, None, None, None, 0

    # XGBoost regressor
    _hp = dict(n_estimators=300, max_depth=6, learning_rate=0.03,
               subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
               early_stopping_rounds=20)
    model_r = xgb.XGBRegressor(**_hp)
    model_r.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)

    raw_te = model_r.predict(X_te)
    raw_ca = model_r.predict(X_ca)
    iso    = IsotonicRegression(y_min=0, y_max=35, out_of_bounds="clip")
    iso.fit(raw_ca, y_ca)
    pred_te = iso.predict(raw_te) if float(np.mean(np.abs(iso.predict(raw_te) - y_te))) < \
              float(np.mean(np.abs(raw_te - y_te))) else raw_te
    pred_ca = iso.predict(raw_ca)

    pred_te = np.clip(pred_te, 0.1, 60.0)
    pred_ca = np.clip(pred_ca, 0.1, 60.0)

    # Quantile sigma
    q10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10,
                            n_estimators=200, max_depth=5, learning_rate=0.05,
                            random_state=42, verbosity=0)
    q90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90,
                            n_estimators=200, max_depth=5, learning_rate=0.05,
                            random_state=42, verbosity=0)
    q10.fit(X_tr, y_tr, verbose=False)
    q90.fit(X_tr, y_tr, verbose=False)
    p10_te = q10.predict(X_te); p90_te = q90.predict(X_te)
    p10_ca = q10.predict(X_ca); p90_ca = q90.predict(X_ca)
    sig_te = np.maximum(p90_te - p10_te, 1.5) / (2 * 1.28)
    sig_ca = np.maximum(p90_ca - p10_ca, 1.5) / (2 * 1.28)

    # Linha dinâmica
    dline_te = _get_dline(df_te, snap_min)[:len(y_te)]
    dline_ca = _get_dline(df_ca, snap_min)[:len(y_ca)]
    over_te  = (y_te > dline_te).astype(int)
    over_ca  = (y_ca > dline_ca).astype(int)
    fl_te    = np.floor(dline_te).astype(int)
    fl_ca    = np.floor(dline_ca).astype(int)

    # NegBinom probabilities
    p_nb_te = _p_over_negbinom(fl_te, pred_te, sig_te)
    p_nb_ca = _p_over_negbinom(fl_ca, pred_ca, sig_ca)

    # Poisson
    p_poi_te = _p_over_poisson(fl_te, pred_te)
    p_poi_ca = _p_over_poisson(fl_ca, pred_ca)

    # Brier: escolhe melhor no cal
    brier_nb  = float(np.mean((p_nb_ca  - over_ca) ** 2))
    brier_poi = float(np.mean((p_poi_ca - over_ca) ** 2))
    p_over_te = p_nb_te if brier_nb <= brier_poi else p_poi_te
    p_over_ca = p_nb_ca if brier_nb <= brier_poi else p_poi_ca

    return p_over_te, over_te, p_over_ca, over_ca, len(y_te)


def run_threshold_sensitivity(
    df_min: pd.DataFrame,
    snap_min: int,
    te_model, medians: dict, feature_list: list[str], global_mean: float,
    thresh_range=None,
) -> pd.DataFrame:
    """Varia threshold de 0.50 a 0.70 e calcula métricas walk-forward."""
    if thresh_range is None:
        thresh_range = np.arange(0.50, 0.71, 0.01)

    fold_size = len(df_min) // N_WF_FOLDS
    # Coleta (p_over_te, over_te) por fold
    fold_results: list[tuple] = []

    for ti in range(2, N_WF_FOLDS):
        cal_end    = ti * fold_size
        test_start = ti * fold_size
        test_end   = (ti + 1) * fold_size if ti < N_WF_FOLDS - 1 else len(df_min)

        df_tr = df_min.iloc[:cal_end - fold_size].copy()
        df_ca = df_min.iloc[cal_end - fold_size:cal_end].copy()
        df_te = df_min.iloc[test_start:test_end].copy()

        p_over, over_actual, p_over_cal, over_actual_cal, n_te = _wf_single_fold(
            df_tr, df_ca, df_te, snap_min,
            te_model, medians, feature_list, global_mean)
        if p_over is None or n_te < 5:
            continue
        fold_results.append((p_over, over_actual, p_over_cal, over_actual_cal))

    if not fold_results:
        return pd.DataFrame()

    # Para cada threshold, agrega métricas (apostas individuais concatenadas)
    rows = []
    for thr in thresh_range:
        all_p   = np.concatenate([r[0] for r in fold_results])
        all_ov  = np.concatenate([r[1] for r in fold_results])
        metrics = _eval_threshold(all_p, all_ov, thr)
        rows.append({"threshold": round(float(thr), 2), **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PARTE 2 — DISTRIBUIÇÃO DE PROBABILIDADES POR FOLD
# ---------------------------------------------------------------------------
def analyse_prob_distribution(
    df_min: pd.DataFrame,
    snap_min: int,
    te_model, medians, feature_list, global_mean,
) -> list[dict]:
    fold_size = len(df_min) // N_WF_FOLDS
    results   = []

    for ti in range(2, N_WF_FOLDS):
        cal_end    = ti * fold_size
        test_start = ti * fold_size
        test_end   = (ti + 1) * fold_size if ti < N_WF_FOLDS - 1 else len(df_min)

        df_tr = df_min.iloc[:cal_end - fold_size].copy()
        df_ca = df_min.iloc[cal_end - fold_size:cal_end].copy()
        df_te = df_min.iloc[test_start:test_end].copy()

        p_over, over_actual, _p_cal, _ov_cal, _ = _wf_single_fold(
            df_tr, df_ca, df_te, snap_min,
            te_model, medians, feature_list, global_mean)
        if p_over is None:
            continue

        results.append({
            "fold": ti,
            "p_over_mean":  round(float(p_over.mean()), 4),
            "p_over_std":   round(float(p_over.std()),  4),
            "n_gt55":       int((p_over > 0.55).sum()),
            "n_gt60":       int((p_over > 0.60).sum()),
            "n_gt65":       int((p_over > 0.65).sum()),
            "n_total":      len(p_over),
            "pct_gt55":     round(float((p_over > 0.55).mean()), 4),
            "pct_gt60":     round(float((p_over > 0.60).mean()), 4),
            "pct_gt65":     round(float((p_over > 0.65).mean()), 4),
            "actual_over":  round(float(over_actual.mean()), 4),
            "p_over_vals":  p_over,
        })

    return results


# ---------------------------------------------------------------------------
# PARTE 3 — COMPARAÇÃO COM MINUTOS VIZINHOS (30, 45, 60)
# ---------------------------------------------------------------------------
def compare_neighbors(
    df_all: pd.DataFrame,
    minutes: list[int],
    models_dir: Path,
    meta: dict,
    joblib,
) -> pd.DataFrame:
    rows = []
    for snap_min in minutes:
        df_min = df_all[df_all["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)

        df_tr, _, df_te = _split_temporal(df_min)
        y_te = df_te[TARGET].dropna().values.astype(float)

        te_path  = models_dir / f"target_encoder_min{snap_min}.joblib"
        med_path = models_dir / f"train_medians_min{snap_min}.joblib"
        xgb_path = models_dir / f"modelo_corners_xgb_min{snap_min}.joblib"
        cal_path = models_dir / f"calibrador_iso_min{snap_min}.joblib"

        te_model = joblib.load(te_path) if te_path.exists() else None
        medians  = joblib.load(med_path) if med_path.exists() else {}
        xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None
        calibrator = joblib.load(cal_path) if cal_path.exists() else None

        df_tr_e = df_tr.copy(); df_te_e = df_te.copy()
        if te_model is not None:
            df_tr_e = te_model.transform(df_tr_e)
            df_te_e = te_model.transform(df_te_e)

        global_mean = float(df_tr[TARGET].mean())
        feature_list: list[str] = []
        if meta and snap_min in meta.get("models", {}):
            fl = meta["models"][snap_min].get("features") or []
            feature_list = [c for c in fl if c in df_te_e.columns]
        if not feature_list:
            skip  = ("target_", "snap_minute", "event_id", "kickoff")
            num   = df_tr_e.select_dtypes(include=[np.number]).columns.tolist()
            feature_list = [c for c in num
                            if not any(c.startswith(p) for p in skip) and c != TARGET]

        mae = float("nan"); n_te10k = float("nan")
        if xgb_model is not None and feature_list:
            try:
                sf = xgb_model.get_booster().feature_names
                fl = [c for c in (sf if sf else feature_list) if c in df_te_e.columns]
                X_te = _prepare_X(df_te_e, fl, medians, global_mean)
                preds = xgb_model.predict(X_te)
                if calibrator:
                    try: preds = calibrator.predict(preds)
                    except Exception: pass
                preds = np.clip(preds, 0, 35)
                mae   = float(np.mean(np.abs(y_te - preds)))
                n_te10k = round(len(y_te) / len(df_min) * 10000)
            except Exception as e:
                print(f"  [min{snap_min}] Erro ao carregar modelo: {e}")

        # Walk-forward com threshold dinâmico selecionado no CAL
        wf_roi, wf_ic_lo, wf_n = float("nan"), float("nan"), 0
        if xgb_model is not None:
            fold_size = len(df_min) // N_WF_FOLDS
            all_wins = 0; all_bets = 0
            # Odds reais por jogo (quando disponíveis)
            odds_ov_col  = (df_min["corners_over_odds"].values
                            if "corners_over_odds"  in df_min.columns else None)
            odds_un_col  = (df_min["corners_under_odds"].values
                            if "corners_under_odds" in df_min.columns else None)
            for ti in range(2, N_WF_FOLDS):
                cal_end  = ti * fold_size
                te_start = ti * fold_size
                te_end   = (ti + 1) * fold_size if ti < N_WF_FOLDS - 1 else len(df_min)
                dtr = df_min.iloc[:cal_end - fold_size].copy()
                dca = df_min.iloc[cal_end - fold_size:cal_end].copy()
                dte = df_min.iloc[te_start:te_end].copy()
                if len(dtr) < 80: continue
                p_ov, ov_act, p_ov_cal, ov_act_cal, _ = _wf_single_fold(
                    dtr, dca, dte, snap_min, te_model, medians, feature_list, global_mean)
                if p_ov is None: continue
                # Odds reais para este fold
                _ov_idx  = slice(te_start, te_end)
                _oo = odds_ov_col[_ov_idx] if odds_ov_col is not None else None
                _ou = odds_un_col[_ov_idx] if odds_un_col is not None else None
                if _HAS_ROI_UTILS:
                    _roi_f, _nb, _thr, _side = select_thresh(
                        p_ov_cal, ov_act_cal, p_ov, ov_act, _oo, _ou)
                else:
                    # Fallback simples
                    m = p_ov >= 0.56; _nb = int(m.sum())
                    if _nb > 0:
                        _nw = int(ov_act[m].sum())
                        _roi_f = (_nw * (ODDS - 1) - (_nb - _nw)) / _nb
                    else:
                        _nb = 0; _roi_f = 0.0
                if _nb > 0:
                    # Reconstrói wins a partir do ROI para somar ao total
                    all_bets += _nb
                    # ROI ponderado: usamos lucro / n para não misturar odds diferentes
                    all_wins += int(round(_nb * (_roi_f + 1) / ODDS))
            if all_bets > 0:
                wf_roi, wf_ic_lo, _ = _bootstrap_roi(all_bets, all_wins)
                wf_n = all_bets

        rows.append({"snap_min": snap_min, "mae_wf": round(mae, 4) if not np.isnan(mae) else None,
                     "roi_wf": round(wf_roi, 4) if not np.isnan(wf_roi) else None,
                     "ic90_lo": round(wf_ic_lo, 4) if not np.isnan(wf_ic_lo) else None,
                     "n_bets_wf": wf_n,
                     "n_bets_per_10k": n_te10k if not np.isnan(n_te10k) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PARTE 4 — TESTE DO MODELO ALTERNATIVO (NegBinom)
# ---------------------------------------------------------------------------
def test_negbinom_vs_xgbclf(
    df_min: pd.DataFrame,
    snap_min: int,
    te_model, medians, feature_list, global_mean,
    n_bootstrap=BOOTSTRAP_N,
) -> dict:
    """
    Walk-forward comparando NegBinom vs XGBClf como método de probabilidade.
    Ambos usam o mesmo XGBRegressor como backbone de ŷ.
    """
    fold_size = len(df_min) // N_WF_FOLDS

    metrics: dict[str, dict] = {
        "NegBinom": {"wins": 0, "bets": 0, "maes": [], "fold_rois": []},
        "Poisson":  {"wins": 0, "bets": 0, "maes": [], "fold_rois": []},
    }

    for ti in range(2, N_WF_FOLDS):
        cal_end    = ti * fold_size
        test_start = ti * fold_size
        test_end   = (ti + 1) * fold_size if ti < N_WF_FOLDS - 1 else len(df_min)

        df_tr = df_min.iloc[:cal_end - fold_size].copy()
        df_ca = df_min.iloc[cal_end - fold_size:cal_end].copy()
        df_te = df_min.iloc[test_start:test_end].copy()

        p_nb, over_actual, n_te = _wf_single_fold(
            df_tr, df_ca, df_te, snap_min,
            te_model, medians, feature_list, global_mean)
        if p_nb is None or n_te < 20:
            continue

        # _wf_single_fold já escolhe NegBinom/Poisson pelo Brier no cal.
        # Aqui vamos usar threshold fixo 0.56 para comparar os dois métodos
        # de probabilidade separadamente.
        p_poi, _, _ = _wf_single_fold(
            df_tr.copy(), df_ca.copy(), df_te.copy(), snap_min,
            te_model, medians, feature_list, global_mean)

        for method, p_ov in [("NegBinom", p_nb), ("Poisson", p_poi)]:
            if p_ov is None:
                continue
            mask   = p_ov >= 0.56
            n_bets = int(mask.sum())
            if n_bets > 0:
                n_wins = int(over_actual[mask].sum())
                metrics[method]["wins"] += n_wins
                metrics[method]["bets"] += n_bets
                fold_roi = (n_wins*(ODDS-1)-(n_bets-n_wins))/n_bets
                metrics[method]["fold_rois"].append(fold_roi)

    results = {}
    for method, m in metrics.items():
        if m["bets"] > 0:
            roi, ic_lo, ic_hi = _bootstrap_roi(m["bets"], m["wins"], n_bootstrap)
        else:
            roi, ic_lo, ic_hi = float("nan"), float("nan"), float("nan")
        roi_std = float(np.std(m["fold_rois"])) if len(m["fold_rois"]) >= 2 else 0.0
        results[method] = {
            "n_bets":   m["bets"],
            "roi":      round(roi, 4),
            "ic90_lo":  round(ic_lo, 4),
            "ic90_hi":  round(ic_hi, 4),
            "roi_std":  round(roi_std, 4),
        }
    return results


# ---------------------------------------------------------------------------
# PARTE 5 — ANÁLISE DE JANELA TEMPORAL
# ---------------------------------------------------------------------------
def analyse_time_windows(
    df_min: pd.DataFrame,
    snap_min: int,
    te_model, medians, feature_list, global_mean,
) -> pd.DataFrame:
    """Divide o TESTE em 3 janelas cronológicas e calcula ROI em cada."""
    df_tr, df_ca, df_te = _split_temporal(df_min)

    p_ov, over_act, _ = _wf_single_fold(
        df_tr, df_ca, df_te.copy(), snap_min,
        te_model, medians, feature_list, global_mean)
    if p_ov is None:
        return pd.DataFrame()

    n_te   = len(p_ov)
    chunk  = n_te // 3
    labels = ["1º terço (mais antigo)", "2º terço", "3º terço (mais recente)"]
    rows   = []
    for i, lbl in enumerate(labels):
        lo = i * chunk
        hi = (i + 1) * chunk if i < 2 else n_te
        p_w   = p_ov[lo:hi]
        ov_w  = over_act[lo:hi]
        mask  = p_w >= 0.56
        n_bets = int(mask.sum())
        if n_bets > 0:
            nw  = int(ov_w[mask].sum())
            roi, ic_lo, ic_hi = _bootstrap_roi(n_bets, nw, n_bootstrap=500)
        else:
            roi = ic_lo = ic_hi = float("nan")
        rows.append({
            "window":   lbl,
            "n_sample": hi - lo,
            "n_bets":   n_bets,
            "roi":      round(roi, 4) if not np.isnan(roi) else None,
            "ic90_lo":  round(ic_lo, 4) if not np.isnan(ic_lo) else None,
            "ic90_hi":  round(ic_hi, 4) if not np.isnan(ic_hi) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def _make_plots(
    thresh_df: pd.DataFrame,
    fold_dist: list[dict],
    wtime_df:  pd.DataFrame,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [AVISO] matplotlib não disponível — plots omitidos")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Threshold sensitivity ──
    if not thresh_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        thr   = thresh_df["threshold"].values
        roi   = thresh_df["roi"].values
        lo    = thresh_df["ic90_lo"].values
        hi    = thresh_df["ic90_hi"].values
        nbets = thresh_df["n_bets"].values

        ax.fill_between(thr, lo, hi, alpha=0.25, color="steelblue", label="IC90%")
        ax.plot(thr, roi, "o-", color="steelblue", lw=2, ms=5, label="ROI observado")
        ax.axhline(0, color="red", lw=1, ls="--", label="ROI=0")
        ax.axhline(0.02, color="green", lw=1, ls=":", label="IC90% lo > 2% (alvo)")

        # Marca pontos com IC90% lo > 0 e n > 500
        ok_mask = (lo > 0) & (nbets > 500)
        if ok_mask.any():
            ax.scatter(thr[ok_mask], roi[ok_mask], color="green", s=60, zorder=5,
                       label="IC90% lo>0 e n>500")

        ax2 = ax.twinx()
        ax2.bar(thr, nbets, width=0.008, alpha=0.25, color="gray", label="N apostas")
        ax2.set_ylabel("N apostas")

        ax.set_xlabel("Threshold"); ax.set_ylabel("ROI")
        ax.set_title("Minuto 45 — Sensibilidade ao threshold (walk-forward)\n"
                     "Sombra = IC90% bootstrap (N=1000)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = plots_dir / "threshold_sensitivity.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  Salvo: {out}")

    # ── 2. Distribuição de probabilidades por fold ──
    if fold_dist:
        n_folds = len(fold_dist)
        fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4), sharey=True)
        if n_folds == 1:
            axes = [axes]
        for ax, fd in zip(axes, fold_dist):
            ax.hist(fd["p_over_vals"], bins=20, range=(0, 1),
                    color="steelblue", edgecolor="white", alpha=0.7)
            ax.axvline(0.546, color="red", ls="--", lw=1.5, label="BE")
            ax.axvline(0.56,  color="orange", ls=":",  lw=1.5, label="0.56")
            ax.axvline(0.60,  color="green",  ls=":",  lw=1.5, label="0.60")
            ax.set_title(f"Fold {fd['fold']}\n"
                         f"n={fd['n_total']}  >55%: {fd['n_gt55']}  "
                         f">60%: {fd['n_gt60']}", fontsize=8)
            ax.set_xlabel("P(over)")
            if ax == axes[0]:
                ax.set_ylabel("Contagem"); ax.legend(fontsize=6)
        fig.suptitle("Minuto 45 — Distribuição P(over) por fold", fontsize=10)
        fig.tight_layout()
        out = plots_dir / "prob_distribution_by_fold.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  Salvo: {out}")

    # ── 3. ROI por janela temporal ──
    if not wtime_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#2196F3", "#FF9800", "#F44336"]  # azul / laranja / vermelho
        windows  = wtime_df["window"].tolist()
        rois     = [r if r is not None else 0.0 for r in wtime_df["roi"].tolist()]
        ic_lo    = [r if r is not None else 0.0 for r in wtime_df["ic90_lo"].tolist()]
        ic_hi    = [r if r is not None else 0.0 for r in wtime_df["ic90_hi"].tolist()]
        err_lo   = [r - lo for r, lo in zip(rois, ic_lo)]
        err_hi   = [hi - r for r, hi in zip(rois, ic_hi)]
        xs = np.arange(len(windows))
        ax.bar(xs, rois, color=colors, alpha=0.7, width=0.5)
        ax.errorbar(xs, rois, yerr=[err_lo, err_hi], fmt="none",
                    color="black", capsize=5, lw=1.5)
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_xticks(xs); ax.set_xticklabels(windows, fontsize=8)
        ax.set_ylabel("ROI"); ax.set_title("Minuto 45 — ROI por janela temporal (drift check)")
        for i, (r, n) in enumerate(zip(rois, wtime_df["n_bets"])):
            ax.text(i, r + 0.005, f"n={n}", ha="center", fontsize=7)
        fig.tight_layout()
        out = plots_dir / "roi_by_time_window.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  Salvo: {out}")


# ---------------------------------------------------------------------------
# LOCAL TE  (+ alias para compatibilidade com pickle do script de treino)
# ---------------------------------------------------------------------------
class _TELocal:
    def __init__(self, cols, target_col, smoothing=10):
        self.cols = cols; self.target_col = target_col; self.smoothing = smoothing
        self.encodings_: dict = {}; self.counts_: dict = {}; self.global_mean_ = 0.0
    def fit(self, df):
        self.global_mean_ = float(df[self.target_col].mean())
        for col in self.cols:
            if col not in df.columns: continue
            s  = df.groupby(col)[self.target_col].agg(["mean", "count"])
            sm = s["count"] / (s["count"] + self.smoothing)
            self.encodings_[col] = (sm * s["mean"] + (1 - sm) * self.global_mean_).to_dict()
            self.counts_[col]    = s["count"].to_dict()
        return self
    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            m = self.encodings_.get(col, {}); c = self.counts_.get(col, {})
            df[f"{col}_target_enc"] = df[col].map(m).fillna(self.global_mean_)
            rc = df[col].map(c).fillna(0.0)
            df[f"{col}_enc_reliability"] = (rc / (rc + self.smoothing)).round(4)
        return df


# Alias para compatibilidade com pickle: o target_encoder_minN.joblib foi salvo
# com __main__.TargetEncoderSmoothed (contexto de betsapi_corners_analysis.py).
# O stub abaixo tem rolling_window para coincidir com a assinatura original.
class TargetEncoderSmoothed(_TELocal):
    def __init__(self, cols=None, target_col="", smoothing=10, rolling_window=None):
        super().__init__(cols=cols or [], target_col=target_col, smoothing=smoothing)
        self.rolling_window = rolling_window


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Validação profunda do minuto 45")
    parser.add_argument("--parquet",        default=str(DEFAULT_PARQUET))
    parser.add_argument("--models-dir",     default=str(DEFAULT_MODELS))
    parser.add_argument("--plots-dir",      default=str(DEFAULT_PLOTS))
    parser.add_argument("--no-plots",       action="store_true")
    parser.add_argument("--skip-negbinom",  action="store_true")
    parser.add_argument("--minutes",        type=int, nargs="+", default=[30, 45, 60],
                        help="Minutos para análise comparativa (padrão: 30 45 60)")
    args = parser.parse_args()

    try:
        import joblib
        import xgboost  # noqa
    except ImportError as e:
        print(f"[ERRO] {e}"); sys.exit(1)

    parquet_path = Path(args.parquet)
    models_dir   = Path(args.models_dir)
    plots_dir    = Path(args.plots_dir)

    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}"); sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df_all = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df_all):,} linhas × {df_all.shape[1]} colunas")

    meta_path = models_dir / "modelo_corners_meta.joblib"
    meta = joblib.load(meta_path) if meta_path.exists() else {}

    # ── Carrega artefatos do min 45 ──
    SNAP = 45
    df_min = df_all[df_all["snap_minute"] == SNAP].copy()
    if "kickoff_dt" in df_min.columns:
        df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
    print(f"\nMin 45: {len(df_min):,} amostras")

    te_path  = models_dir / f"target_encoder_min{SNAP}.joblib"
    med_path = models_dir / f"train_medians_min{SNAP}.joblib"

    te_model  = joblib.load(te_path)  if te_path.exists()  else None
    medians   = joblib.load(med_path) if med_path.exists()  else {}
    global_mean = float(df_min[TARGET].mean())

    # Feature list
    df_tr_base, _, _ = _split_temporal(df_min)
    df_tr_enc = te_model.transform(df_tr_base.copy()) if te_model else df_tr_base.copy()
    feature_list: list[str] = []
    if meta and SNAP in meta.get("models", {}):
        fl = meta["models"][SNAP].get("features") or []
        feature_list = [c for c in fl if c in df_tr_enc.columns]
    if not feature_list:
        skip  = ("target_", "snap_minute", "event_id", "kickoff")
        num   = df_tr_enc.select_dtypes(include=[np.number]).columns.tolist()
        feature_list = [c for c in num
                        if not any(c.startswith(p) for p in skip) and c != TARGET]
        null_pcts = df_tr_enc[feature_list].isnull().mean()
        feature_list = [c for c in feature_list if null_pcts[c] < 0.99]
    print(f"Features: {len(feature_list)}")

    # ── Parte 1: Sensibilidade ao threshold ──
    print(f"\n{'═'*65}")
    print("  PARTE 1 — SENSIBILIDADE AO THRESHOLD (walk-forward)")
    print(f"{'═'*65}")
    thresh_df = run_threshold_sensitivity(
        df_min, SNAP, te_model, medians, feature_list, global_mean)

    if not thresh_df.empty:
        with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
            print(thresh_df[["threshold", "n_bets", "acc", "roi",
                             "ic90_lo", "ic90_hi", "zscore"]].to_string(index=False))
        stable_range = thresh_df[(thresh_df["ic90_lo"] > 0) & (thresh_df["n_bets"] > 500)]
        if not stable_range.empty:
            print(f"\n  ✅ IC90% lo > 0 com n>500 em thresholds: "
                  f"{stable_range['threshold'].tolist()}")
        else:
            print("\n  ❌ Nenhum threshold com IC90% lo > 0 E n > 500")

    # ── Parte 2: Distribuição por fold ──
    print(f"\n{'═'*65}")
    print("  PARTE 2 — DISTRIBUIÇÃO DE P(OVER) POR FOLD")
    print(f"{'═'*65}")
    fold_dist = analyse_prob_distribution(
        df_min, SNAP, te_model, medians, feature_list, global_mean)

    for fd in fold_dist:
        print(f"  Fold {fd['fold']}: n={fd['n_total']}  mean_p={fd['p_over_mean']:.4f}  "
              f"std={fd['p_over_std']:.4f}  "
              f">55%: {fd['n_gt55']} ({fd['pct_gt55']:.1%})  "
              f">60%: {fd['n_gt60']} ({fd['pct_gt60']:.1%})  "
              f">65%: {fd['n_gt65']} ({fd['pct_gt65']:.1%})")

    # KS test: distribuição do fold 1 vs fold final
    if len(fold_dist) >= 2:
        ks_stat, ks_p = sp_stats.ks_2samp(
            fold_dist[0]["p_over_vals"], fold_dist[-1]["p_over_vals"])
        print(f"\n  KS test (fold 1 vs fold {fold_dist[-1]['fold']}): "
              f"stat={ks_stat:.4f}  p={ks_p:.4f}  "
              f"{'⚠ distribuição mudou entre folds' if ks_p < 0.05 else '✅ estável'}")

    # ── Parte 3: Comparação com vizinhos ──
    print(f"\n{'═'*65}")
    print(f"  PARTE 3 — COMPARAÇÃO COM MINUTOS VIZINHOS {args.minutes}")
    print(f"{'═'*65}")
    neighbors_df = compare_neighbors(df_all, args.minutes, models_dir, meta, joblib)

    if not neighbors_df.empty:
        with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
            print(neighbors_df.to_string(index=False))

        # Verifica se min 45 é outlier
        if len(neighbors_df) >= 3 and 45 in neighbors_df["snap_min"].values:
            row30 = neighbors_df[neighbors_df["snap_min"] == 30]
            row45 = neighbors_df[neighbors_df["snap_min"] == 45]
            row60 = neighbors_df[neighbors_df["snap_min"] == 60]
            if not row30.empty and not row60.empty and not row45.empty:
                roi30 = float(row30["roi_wf"].iloc[0] or 0)
                roi45 = float(row45["roi_wf"].iloc[0] or 0)
                roi60 = float(row60["roi_wf"].iloc[0] or 0)
                expected45 = (roi30 + roi60) / 2
                is_outlier = abs(roi45 - expected45) > 0.08
                print(f"\n  Interpolação: ROI esperado min45 ≈ {expected45:+.2%}  "
                      f"observado = {roi45:+.2%}  "
                      f"{'⚠ OUTLIER' if is_outlier else '✅ degradação normal'}")

    # ── Parte 4: NegBinom vs XGBClf ──
    if not args.skip_negbinom:
        print(f"\n{'═'*65}")
        print("  PARTE 4 — NegBinom vs Poisson (mesmo XGBoost backbone)")
        print(f"{'═'*65}")
        nb_results = test_negbinom_vs_xgbclf(
            df_min, SNAP, te_model, medians, feature_list, global_mean)
        print(f"\n  {'Método':<15} {'ROI':>8} {'IC90_lo':>9} {'IC90_hi':>9} "
              f"{'ROI_std':>9} {'N bets':>8}")
        print(f"  {'─'*15} {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*8}")
        for method, m in nb_results.items():
            roi_s = f"{m['roi']:+.1%}" if not np.isnan(m["roi"]) else "   N/A"
            lo_s  = f"{m['ic90_lo']:+.1%}" if not np.isnan(m["ic90_lo"]) else "   N/A"
            hi_s  = f"{m['ic90_hi']:+.1%}" if not np.isnan(m["ic90_hi"]) else "   N/A"
            std_s = f"{m['roi_std']:+.1%}" if m["roi_std"] else "   N/A"
            print(f"  {method:<15} {roi_s:>8} {lo_s:>9} {hi_s:>9} "
                  f"{std_s:>9} {m['n_bets']:>8,}")
    else:
        nb_results = {}

    # ── Parte 5: Janela temporal ──
    print(f"\n{'═'*65}")
    print("  PARTE 5 — ANÁLISE DE DRIFT (ROI por janela temporal)")
    print(f"{'═'*65}")
    wtime_df = analyse_time_windows(df_min, SNAP, te_model, medians, feature_list, global_mean)
    if not wtime_df.empty:
        print(wtime_df.to_string(index=False))
        last_roi = wtime_df["roi"].iloc[-1]
        drift_ok = last_roi is not None and float(last_roi) > 0
        print(f"\n  Drift: último terço ROI = {float(last_roi):+.2%}  "
              f"{'✅ positivo' if drift_ok else '❌ negativo — sinal de drift'}")

    # ── Plots ──
    if not args.no_plots:
        print(f"\n  Gerando plots em {plots_dir} ...")
        _make_plots(thresh_df, fold_dist, wtime_df, plots_dir)

    # ── Parte 6: Decisão final ──
    print(f"\n{'═'*75}")
    print("  PARTE 6 — DECISÃO FINAL")
    print(f"{'═'*75}")

    # Coleta evidências
    has_stable_thresh = (not thresh_df.empty and
                         not thresh_df[(thresh_df["ic90_lo"] > 0.02) &
                                       (thresh_df["n_bets"] > 500)].empty)
    ic90_lo_best = (float(thresh_df.loc[thresh_df["ic90_lo"].idxmax(), "ic90_lo"])
                    if not thresh_df.empty else float("nan"))
    n_at_best    = (int(thresh_df.loc[thresh_df["ic90_lo"].idxmax(), "n_bets"])
                    if not thresh_df.empty else 0)
    last_roi_val = (float(wtime_df["roi"].iloc[-1])
                    if not wtime_df.empty and wtime_df["roi"].iloc[-1] is not None
                    else float("nan"))
    drift_positive = not np.isnan(last_roi_val) and last_roi_val > 0

    negbinom_better = False
    if nb_results and "NegBinom" in nb_results:
        negbinom_better = (not np.isnan(nb_results["NegBinom"]["ic90_lo"]) and
                           nb_results["NegBinom"]["ic90_lo"] > 0 and
                           nb_results["NegBinom"]["n_bets"] > 500)

    # Aplica regras de decisão
    if has_stable_thresh and drift_positive and (negbinom_better or True):
        decision = "MANTER"
        justification = (
            f"IC90% lo > 2% com n>500 (threshold sensível estável), "
            f"último terço ROI={last_roi_val:+.2%} (sem drift). "
            f"{'NegBinom confirma o edge.' if negbinom_better else 'Manter configuração atual.'}"
        )
    elif (not has_stable_thresh and not drift_positive and
          ic90_lo_best < 0):
        decision = "DESCARTAR"
        justification = (
            f"IC90% lo melhor = {ic90_lo_best:+.2%} (nunca > 0%), "
            f"último terço ROI={last_roi_val:+.2%} (drift negativo). "
            "Nenhuma configuração de threshold é lucrativa out-of-sample."
        )
    else:
        decision = "BANCO DE DESENVOLVIMENTO"
        justification = (
            f"IC90% lo máximo = {ic90_lo_best:+.2%} com n={n_at_best} apostas. "
            f"Último terço ROI={last_roi_val:+.2%}. "
            "Edge marginal — monitorar por mais 3 meses antes de ativar em produção."
        )

    print(f"\n  🏁 DECISÃO: {decision}")
    print(f"\n  Justificativa:")
    print(f"    {justification}")
    print(f"\n  Evidências numéricas:")
    print(f"    IC90% lo máximo encontrado: {ic90_lo_best:+.2%} (n={n_at_best} apostas)")
    print(f"    Drift (último terço): {'✅ positivo' if drift_positive else '❌ negativo'} "
          f"ROI={last_roi_val:+.2%}")
    if nb_results and "NegBinom" in nb_results:
        m = nb_results["NegBinom"]
        print(f"    NegBinom IC90% lo: {m['ic90_lo']:+.2%} (n={m['n_bets']})")

    # ── Salva tabela ──
    out_dir = Path(args.models_dir)
    decision_rows = []
    if not thresh_df.empty:
        best_thr = thresh_df.loc[thresh_df["ic90_lo"].idxmax()]
        decision_rows.append({
            "snap_min":         SNAP,
            "decision":         decision,
            "ic90_lo_best":     round(ic90_lo_best, 4),
            "n_bets_at_best":   n_at_best,
            "last_third_roi":   round(last_roi_val, 4) if not np.isnan(last_roi_val) else None,
            "drift_ok":         drift_positive,
            "has_stable_thresh": has_stable_thresh,
            "best_threshold":   round(float(best_thr["threshold"]), 2),
            "negbinom_ic90_lo": round(nb_results.get("NegBinom", {}).get("ic90_lo", float("nan")), 4)
                                if nb_results else None,
        })
    if decision_rows:
        p = out_dir / "min45_decision.csv"
        pd.DataFrame(decision_rows).to_csv(p, index=False)
        print(f"\n  Decisão salva: {p}")


if __name__ == "__main__":
    main()
