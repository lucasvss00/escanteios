"""
recalibrate_ngboost.py
======================
Recalibração isotônica das probabilidades NGBoost + comparação walk-forward.

Fluxo:
  1. Para cada (snap_min, linha), ajusta IsotonicRegression no conjunto de
     calibração: p_raw_ngb → frequência empírica real.
  2. Salva calibradores em dados_escanteios/isotonic_calibrators_ngb.joblib.
  3. (Opcional) Walk-forward: compara ROI / Brier / Accuracy NGBoost raw
     vs calibrado vs Poisson approx.
  4. Imprime decisão e snippet de integração no pipeline principal.

Uso:
    python recalibrate_ngboost.py
    python recalibrate_ngboost.py --walk-forward
    python recalibrate_ngboost.py --walk-forward --n-estimators 200
    python recalibrate_ngboost.py --minutes 60 75
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import lognorm as sp_lognorm
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

# Importa utilitários de ROI compartilhados (mesma lógica do pipeline principal)
try:
    from _roi_utils import select_thresh, ODDS_OVER, ODDS_UNDER, BREAKEVEN, MIN_EDGE
    _HAS_ROI_UTILS = True
except ImportError:
    _HAS_ROI_UTILS = False

# ---------------------------------------------------------------------------
# PICKLE COMPAT — betsapi_corners_analysis.py salva TargetEncoderSmoothed
# com __main__.TargetEncoderSmoothed; este stub permite carregar o .joblib
# ---------------------------------------------------------------------------
class TargetEncoderSmoothed:
    """Stub de compatibilidade para joblib.load do target encoder treinado."""
    def __init__(self, cols=None, target_col="", smoothing=10, rolling_window=None):
        self.cols = cols or []
        self.target_col = target_col
        self.smoothing = smoothing
        self.rolling_window = rolling_window
        self.encodings_: dict = {}
        self.counts_:    dict = {}
        self.global_mean_: float = 0.0

    def transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as _pd
        df = df.copy()
        for col in self.cols:
            m = self.encodings_.get(col, {})
            c = self.counts_.get(col, {})
            df[f"{col}_target_enc"] = df[col].map(m).fillna(self.global_mean_)
            rc = df[col].map(c).fillna(0.0)
            df[f"{col}_enc_reliability"] = (rc / (rc + self.smoothing)).round(4)
        return df


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET  = Path("dados_escanteios/features_ml.parquet")
DEFAULT_MODELS   = Path("dados_escanteios")
DEFAULT_CAL_OUT  = Path("dados_escanteios/isotonic_calibrators_ngb.joblib")
SNAPSHOT_MINUTES = [15, 30, 45, 60, 75]
TARGET           = "target_corners_total"
ODDS             = 1.83
MIN_BIN_SAMPLES  = 20     # amostras mínimas por bin para IsotonicRegression
ECE_THRESHOLD    = 0.06   # ECE acima disso → recalibração obrigatória
ECE_IMPROVE_MIN  = 0.30   # melhora mínima de ECE para adotar calibração


# ---------------------------------------------------------------------------
# HELPERS (mesmos de calibration_diagnosis.py — standalone)
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


def _prepare_X(
    df: pd.DataFrame,
    feature_list: list[str],
    medians: dict,
    global_mean: float,
) -> np.ndarray:
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


def _dynamic_line(df: pd.DataFrame, snap_min: int) -> np.ndarray:
    rem = 90 - snap_min
    csf = (df["corners_total_so_far"].values
           if "corners_total_so_far" in df.columns else np.zeros(len(df)))
    rate = (np.where(df["league_avg_corners"].isna(),
                     11.0 / 90.0, df["league_avg_corners"].values / 90.0)
            if "league_avg_corners" in df.columns
            else np.full(len(df), 11.0 / 90.0))
    return csf + rem * rate


def _lognorm_params(dist_params: dict) -> tuple[np.ndarray, np.ndarray]:
    s     = np.clip(np.array(dist_params["s"],     dtype=float), 1e-6, 10.0)
    scale = np.clip(np.array(dist_params["scale"], dtype=float), 1e-6, 1e6)
    return s, scale


def _lognorm_mean(s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(scale * np.exp(s ** 2 / 2), 0.01, 60.0)


def _p_over_lognorm(line: np.ndarray, s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    fl = np.floor(line).astype(float) + 0.5
    x  = np.maximum(fl, 1e-9)
    return np.clip(1.0 - sp_lognorm.cdf(x, s=s, scale=scale), 1e-6, 1 - 1e-6)


def _p_over_poisson(line: np.ndarray, mu: np.ndarray) -> np.ndarray:
    fl = np.floor(line).astype(int)
    return np.clip(
        np.array([1.0 - sp_poisson.cdf(int(fl[i]), mu=max(float(mu[i]), 0.01))
                  for i in range(len(mu))]),
        1e-6, 1 - 1e-6,
    )


def _ece(p_pred: np.ndarray, y_bin: np.ndarray, n_bins: int = 20) -> float:
    bins  = np.linspace(0, 1, n_bins + 1)
    total = len(p_pred)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / total * abs(p_pred[mask].mean() - y_bin[mask].mean())
    return round(ece, 6)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _roi(p: np.ndarray, y: np.ndarray,
         thresh: float = 0.5, odds: float = ODDS) -> tuple[float, int]:
    mask   = p > thresh
    n_bets = int(mask.sum())
    if n_bets == 0:
        return float("nan"), 0
    wins   = y[mask]
    profit = (wins * (odds - 1) - (1 - wins)).sum()
    return float(profit / n_bets), n_bets


# ---------------------------------------------------------------------------
# ISOTONIC CALIBRATION
# ---------------------------------------------------------------------------
def fit_isotonic(
    p_raw: np.ndarray,
    y_binary: np.ndarray,
) -> object:
    """
    Ajusta IsotonicRegression mapeando p_raw → frequência empírica.
    Requer sklearn.
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p_raw, y_binary)
    return iso


def calibrate(iso, p_raw: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(p_raw), 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# CARREGA MODELOS E REBUILD X_TEST
# ---------------------------------------------------------------------------
def load_split_for_minute(
    df_all: pd.DataFrame,
    snap_min: int,
    models_dir: Path,
    meta: dict,
) -> Optional[tuple]:
    """
    Retorna (X_cal, X_te, y_cal, y_te, df_ca, df_te, ngb_model, feature_list).
    None se não for possível.
    """
    try:
        import joblib
    except ImportError:
        return None

    ngb_path = models_dir / f"modelo_corners_ngb_min{snap_min}.joblib"
    if not ngb_path.exists():
        print(f"  [AVISO] NGBoost não encontrado para min{snap_min}: {ngb_path}")
        return None

    ngb_model = joblib.load(ngb_path)

    df_min = df_all[df_all["snap_minute"] == snap_min].copy()
    if "kickoff_dt" in df_min.columns:
        df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
    df_tr, df_ca, df_te = _split_temporal(df_min)

    te_path = models_dir / f"target_encoder_min{snap_min}.joblib"
    if te_path.exists():
        te = joblib.load(te_path)
        df_tr = te.transform(df_tr)
        df_ca = te.transform(df_ca)
        df_te = te.transform(df_te)

    med_path = models_dir / f"train_medians_min{snap_min}.joblib"
    train_medians: dict = joblib.load(med_path) if med_path.exists() else {}
    global_mean = float(df_tr[TARGET].mean()) if TARGET in df_tr.columns else 11.0

    feature_list: list[str] = []
    if meta and snap_min in meta.get("models", {}):
        fl = meta["models"][snap_min].get("features") or []
        feature_list = [c for c in fl if c in df_te.columns]
    if not feature_list:
        skip = ("target_", "snap_minute", "event_id", "kickoff")
        num  = df_tr.select_dtypes(include=[np.number]).columns.tolist()
        feature_list = [c for c in num
                        if not any(c.startswith(p) for p in skip) and c != TARGET]
        null_pcts    = df_tr[feature_list].isnull().mean()
        feature_list = [c for c in feature_list if null_pcts[c] < 0.99]

    X_cal = _prepare_X(df_ca, feature_list, train_medians, global_mean)
    X_te  = _prepare_X(df_te, feature_list, train_medians, global_mean)
    y_cal = df_ca[TARGET].dropna().values.astype(float)
    y_te  = df_te[TARGET].dropna().values.astype(float)

    return X_cal, X_te, y_cal, y_te, df_ca, df_te, ngb_model, feature_list


# ---------------------------------------------------------------------------
# CALIBRAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
def run_calibration(
    df_all: pd.DataFrame,
    models_dir: Path,
    cal_out_path: Path,
    meta: dict,
    minutes: list[int],
) -> dict:
    """
    Ajusta e salva calibradores isotônicos por (snap_min).
    Retorna dict de diagnóstico.
    """
    try:
        import joblib
        from sklearn.isotonic import IsotonicRegression  # noqa
    except ImportError as e:
        print(f"[ERRO] {e}")
        return {}

    calibrators: dict[int, dict] = {}
    report: list[dict] = []

    for snap_min in minutes:
        print(f"\n{'─'*60}")
        print(f"  Minuto {snap_min}")
        print(f"{'─'*60}")

        res = load_split_for_minute(df_all, snap_min, models_dir, meta)
        if res is None:
            continue
        X_cal, X_te, y_cal, y_te, df_ca, df_te, ngb_model, feat_list = res

        print(f"  Calculando pred_dist (cal={len(X_cal)}, test={len(X_te)})...")
        try:
            dist_cal  = ngb_model.pred_dist(X_cal)
            dist_te   = ngb_model.pred_dist(X_te)
        except Exception as e:
            print(f"  [ERRO] pred_dist: {e}")
            continue

        s_cal, sc_cal = _lognorm_params(dist_cal.params)
        s_te,  sc_te  = _lognorm_params(dist_te.params)
        mu_cal = _lognorm_mean(s_cal, sc_cal)
        mu_te  = _lognorm_mean(s_te,  sc_te)

        lines_cal = _dynamic_line(df_ca, snap_min)[:len(y_cal)]
        lines_te  = _dynamic_line(df_te, snap_min)[:len(y_te)]

        over_cal = (y_cal > lines_cal).astype(float)
        over_te  = (y_te  > lines_te).astype(float)

        # Probabilidades raw (LogNorm e Poisson)
        p_lognorm_cal = _p_over_lognorm(lines_cal, s_cal, sc_cal)
        p_lognorm_te  = _p_over_lognorm(lines_te,  s_te,  sc_te)
        p_poisson_cal = _p_over_poisson(lines_cal, mu_cal)
        p_poisson_te  = _p_over_poisson(lines_te,  mu_te)

        # Métricas raw
        ece_ln_raw    = _ece(p_lognorm_te,  over_te)
        ece_poi_raw   = _ece(p_poisson_te,  over_te)
        brier_ln_raw  = _brier(p_lognorm_te, over_te)
        brier_poi_raw = _brier(p_poisson_te, over_te)
        roi_ln_raw,   nb_ln  = _roi(p_lognorm_te,  over_te)
        roi_poi_raw,  nb_poi = _roi(p_poisson_te, over_te)

        print(f"  Raw LogNorm  — ECE={ece_ln_raw:.5f}  Brier={brier_ln_raw:.5f}  "
              f"ROI={roi_ln_raw:+.2%}  bets={nb_ln}")
        print(f"  Raw Poisson  — ECE={ece_poi_raw:.5f}  Brier={brier_poi_raw:.5f}  "
              f"ROI={roi_poi_raw:+.2%}  bets={nb_poi}")

        # ── Fit isotonic no CAL ──
        iso_lognorm = fit_isotonic(p_lognorm_cal, over_cal)
        iso_poisson = fit_isotonic(p_poisson_cal, over_cal)

        p_lognorm_cal_recal = calibrate(iso_lognorm, p_lognorm_cal)
        p_poisson_cal_recal = calibrate(iso_poisson, p_poisson_cal)

        # Sanidade: ECE no CAL antes/depois
        ece_ln_cal_raw   = _ece(p_lognorm_cal, over_cal)
        ece_ln_cal_recal = _ece(p_lognorm_cal_recal, over_cal)
        print(f"  ECE LogNorm no CAL: raw={ece_ln_cal_raw:.5f} → recal={ece_ln_cal_recal:.5f} "
              f"(nota: treinado no cal, overfits)")

        # ── Avalia no TEST ──
        p_lognorm_te_recal = calibrate(iso_lognorm, p_lognorm_te)
        p_poisson_te_recal = calibrate(iso_poisson, p_poisson_te)

        ece_ln_recal    = _ece(p_lognorm_te_recal,  over_te)
        ece_poi_recal   = _ece(p_poisson_te_recal,  over_te)
        brier_ln_recal  = _brier(p_lognorm_te_recal, over_te)
        brier_poi_recal = _brier(p_poisson_te_recal, over_te)
        roi_ln_recal,   nb_lr  = _roi(p_lognorm_te_recal,  over_te)
        roi_poi_recal,  nb_pr  = _roi(p_poisson_te_recal,  over_te)

        ece_improve_ln  = (ece_ln_raw  - ece_ln_recal)  / max(ece_ln_raw, 1e-9)
        ece_improve_poi = (ece_poi_raw - ece_poi_recal) / max(ece_poi_raw, 1e-9)

        print(f"  Recal LogNorm — ECE={ece_ln_recal:.5f}  Brier={brier_ln_recal:.5f}  "
              f"ROI={roi_ln_recal:+.2%}  bets={nb_lr}  "
              f"(ECE melhora {ece_improve_ln:+.1%})")
        print(f"  Recal Poisson — ECE={ece_poi_recal:.5f}  Brier={brier_poi_recal:.5f}  "
              f"ROI={roi_poi_recal:+.2%}  bets={nb_pr}  "
              f"(ECE melhora {ece_improve_poi:+.1%})")

        # Decide qual versão adotar
        best_ece  = min(ece_ln_raw, ece_poi_raw, ece_ln_recal, ece_poi_recal)
        if best_ece == ece_ln_recal:
            adopted = "lognorm_recalibrated"
        elif best_ece == ece_poi_recal:
            adopted = "poisson_recalibrated"
        elif best_ece == ece_ln_raw:
            adopted = "lognorm_raw"
        else:
            adopted = "poisson_raw"

        calibrators[snap_min] = {
            "iso_lognorm": iso_lognorm,
            "iso_poisson": iso_poisson,
            "adopted":     adopted,
            "ece_lognorm_raw":    ece_ln_raw,
            "ece_lognorm_recal":  ece_ln_recal,
            "ece_poisson_raw":    ece_poi_raw,
            "ece_poisson_recal":  ece_poi_recal,
        }

        report.append({
            "snap_min":          snap_min,
            "ece_lognorm_raw":   ece_ln_raw,
            "ece_lognorm_recal": ece_ln_recal,
            "ece_poisson_raw":   ece_poi_raw,
            "ece_poisson_recal": ece_poi_recal,
            "brier_ln_raw":      brier_ln_raw,
            "brier_ln_recal":    brier_ln_recal,
            "roi_lognorm_raw":   roi_ln_raw,
            "roi_lognorm_recal": roi_ln_recal,
            "roi_poisson_raw":   roi_poi_raw,
            "roi_poisson_recal": roi_poi_recal,
            "bets_lognorm_raw":  nb_ln,
            "bets_lognorm_recal": nb_lr,
            "adopted":           adopted,
            "ece_improve_lognorm": round(ece_improve_ln, 4),
            "ece_improve_poisson": round(ece_improve_poi, 4),
        })

    # Salva calibradores
    import joblib
    cal_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, cal_out_path)
    print(f"\n  Calibradores salvos: {cal_out_path}")

    return {"calibrators": calibrators, "report": pd.DataFrame(report)}


# ---------------------------------------------------------------------------
# WALK-FORWARD COMPARISON
# ---------------------------------------------------------------------------
def run_walk_forward(
    df_all: pd.DataFrame,
    models_dir: Path,
    meta: dict,
    minutes: list[int],
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Walk-forward comparando NGBoost raw vs calibrado vs Poisson."""
    try:
        import joblib
        from ngboost import NGBRegressor
        from ngboost.distns import LogNormal as NGBLogNormal
        from ngboost.scores import CRPScore
    except ImportError as e:
        print(f"[ERRO] Dependência faltando: {e}")
        return pd.DataFrame()

    N_FOLDS   = 7
    all_rows: list[dict] = []

    for snap_min in minutes:
        print(f"\n{'═'*70}")
        print(f"  Walk-forward — Minuto {snap_min}")
        print(f"{'═'*70}")

        df_min = df_all[df_all["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
        if len(df_min) < 500:
            continue

        fold_size = len(df_min) // N_FOLDS

        # Feature list (do metadata ou inferida)
        te_path  = models_dir / f"target_encoder_min{snap_min}.joblib"
        med_path = models_dir / f"train_medians_min{snap_min}.joblib"

        wf_rows: dict[str, list] = {
            "raw_lognorm": [], "recal_lognorm": [], "poisson_approx": []
        }
        bets_per_method: dict[str, list] = {k: [] for k in wf_rows}

        for ti in range(2, N_FOLDS):
            cal_start  = (ti - 1) * fold_size
            cal_end    = ti * fold_size
            test_start = ti * fold_size
            test_end   = (ti + 1) * fold_size if ti < N_FOLDS - 1 else len(df_min)

            df_tr = df_min.iloc[:cal_start].copy()
            df_ca = df_min.iloc[cal_start:cal_end].copy()
            df_te = df_min.iloc[test_start:test_end].copy()

            if len(df_tr) < 80 or len(df_ca) < 20 or len(df_te) < 20:
                continue

            # Target encoding fit no treino
            if te_path.exists():
                te = joblib.load(te_path)
                # Recria TE apenas no treino deste fold (não usa o TE do treino completo
                # para evitar leakage do fold)
                from recalibrate_ngboost import _TargetEncoderSmoothed_local as _TES
                _te_wf = _TES(te.cols, TARGET, te.smoothing)
                _te_wf.fit(df_tr)
                df_tr = _te_wf.transform(df_tr)
                df_ca = _te_wf.transform(df_ca)
                df_te = _te_wf.transform(df_te)

            # Feature list
            skip  = ("target_", "snap_minute", "event_id", "kickoff")
            num   = df_tr.select_dtypes(include=[np.number]).columns.tolist()
            feats = [c for c in num if not any(c.startswith(p) for p in skip) and c != TARGET]
            null_pcts = df_tr[feats].isnull().mean()
            feats = [c for c in feats if null_pcts[c] < 0.99]

            if not feats:
                continue

            global_mean = float(df_tr[TARGET].mean())
            train_med   = {c: df_tr[c].median() for c in feats
                           if c.startswith(("hist_", "league_", "h2h_")) or c.endswith("_target_enc")}

            X_tr = _prepare_X(df_tr, feats, train_med, global_mean)
            X_ca = _prepare_X(df_ca, feats, train_med, global_mean)
            X_te = _prepare_X(df_te, feats, train_med, global_mean)
            y_tr = np.maximum(df_tr[TARGET].values.astype(float), 0.5)
            y_ca = np.maximum(df_ca[TARGET].values.astype(float), 0.5)
            y_te_real = df_te[TARGET].values.astype(float)

            # Treina NGBoost neste fold
            print(f"    fold {ti}/{N_FOLDS-1}: treino={len(X_tr)} cal={len(X_ca)} test={len(X_te)}")
            try:
                ngb = NGBRegressor(
                    Dist=NGBLogNormal, Score=CRPScore,
                    n_estimators=n_estimators, learning_rate=0.03,
                    minibatch_frac=0.8, verbose=False,
                    random_state=42, natural_gradient=True,
                )
                ngb.fit(X_tr, y_tr, X_val=X_ca, Y_val=y_ca, early_stopping_rounds=20)
            except Exception as e:
                print(f"      NGBoost treino falhou: {e}")
                continue

            dist_ca = ngb.pred_dist(X_ca)
            dist_te = ngb.pred_dist(X_te)

            s_ca, sc_ca = _lognorm_params(dist_ca.params)
            s_te, sc_te = _lognorm_params(dist_te.params)
            mu_ca = _lognorm_mean(s_ca, sc_ca)
            mu_te = _lognorm_mean(s_te, sc_te)

            lines_ca = _dynamic_line(df_ca, snap_min)[:len(y_ca)]
            lines_te = _dynamic_line(df_te, snap_min)[:len(y_te_real)]

            over_ca = (df_ca[TARGET].values.astype(float) > lines_ca).astype(float)
            over_te = (y_te_real > lines_te).astype(float)

            p_ln_ca   = _p_over_lognorm(lines_ca, s_ca, sc_ca)
            p_ln_te   = _p_over_lognorm(lines_te, s_te, sc_te)
            p_poi_te  = _p_over_poisson(lines_te, mu_te)

            # Isotonic fit no cal
            iso_wf = fit_isotonic(p_ln_ca, over_ca)
            p_ln_te_recal = calibrate(iso_wf, p_ln_te)

            for method, p_te in [
                ("raw_lognorm",    p_ln_te),
                ("recal_lognorm",  p_ln_te_recal),
                ("poisson_approx", p_poi_te),
            ]:
                brier = _brier(p_te, over_te)
                roi_v, n_bets = _roi(p_te, over_te)
                wf_rows[method].append({
                    "brier": brier,
                    "roi":   roi_v if not np.isnan(roi_v) else 0.0,
                    "n_bets": n_bets,
                    "ece":   _ece(p_te, over_te),
                })
                bets_per_method[method].append(n_bets)
                roe_str = f"{roi_v:+.1%}" if not np.isnan(roi_v) else "  N/A"
                print(f"      {method:<20} Brier={brier:.5f}  ROI={roe_str}  bets={n_bets}")

        # Agrega folds
        for method, folds in wf_rows.items():
            if not folds:
                continue
            fold_df = pd.DataFrame(folds)
            roi_arr = fold_df["roi"].values
            all_rows.append({
                "snap_min":    snap_min,
                "method":      method,
                "brier_mean":  round(float(fold_df["brier"].mean()), 5),
                "brier_std":   round(float(fold_df["brier"].std()),  5),
                "ece_mean":    round(float(fold_df["ece"].mean()),    5),
                "roi_mean":    round(float(roi_arr.mean()),           4),
                "roi_std":     round(float(roi_arr.std()),            4),
                "roi_ic90_lo": round(float(np.percentile(roi_arr, 5)),  4),
                "roi_ic90_hi": round(float(np.percentile(roi_arr, 95)), 4),
                "n_bets_total": int(sum(bets_per_method[method])),
                "n_folds":     len(folds),
            })

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# DECISÃO FINAL
# ---------------------------------------------------------------------------
def print_decision(report_df: pd.DataFrame, wf_df: Optional[pd.DataFrame]) -> None:
    print("\n" + "=" * 90)
    print("  RELATÓRIO DE CALIBRAÇÃO — NGBoost LogNormal")
    print("=" * 90)

    if not report_df.empty:
        cols = ["snap_min", "ece_lognorm_raw", "ece_lognorm_recal",
                "ece_poisson_raw", "ece_poisson_recal",
                "roi_lognorm_raw", "roi_lognorm_recal", "adopted"]
        with pd.option_context("display.width", 160, "display.float_format", "{:.5f}".format):
            print(report_df[[c for c in cols if c in report_df.columns]].to_string(index=False))

    if wf_df is not None and not wf_df.empty:
        print("\n  WALK-FORWARD COMPARISON")
        print("─" * 90)
        for snap_min in wf_df["snap_min"].unique():
            sub = wf_df[wf_df["snap_min"] == snap_min]
            print(f"\n  Minuto {snap_min}:")
            print(f"  {'Método':<25} {'ECE':>8} {'Brier':>8} {'ROI':>9} "
                  f"{'ROI_std':>8} {'IC90_lo':>8} {'IC90_hi':>8} {'Bets':>7}")
            print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*9} "
                  f"{'─'*8} {'─'*8} {'─'*8} {'─'*7}")
            for _, row in sub.iterrows():
                print(f"  {row['method']:<25} {row['ece_mean']:>8.5f} "
                      f"{row['brier_mean']:>8.5f} {row['roi_mean']:>+8.1%} "
                      f"{row['roi_std']:>+7.1%} "
                      f"{row['roi_ic90_lo']:>+7.1%} {row['roi_ic90_hi']:>+7.1%} "
                      f"{int(row['n_bets_total']):>7,}")

    print("\n" + "=" * 90)
    print("  DECISÃO POR MINUTO")
    print("=" * 90)

    decisions: list[str] = []
    if not report_df.empty:
        for _, row in report_df.iterrows():
            m = int(row["snap_min"])
            ece_raw  = float(row.get("ece_lognorm_raw", 0))
            ece_rcal = float(row.get("ece_lognorm_recal", 0))
            ece_imp  = (ece_raw - ece_rcal) / max(ece_raw, 1e-9)

            if ece_raw < ECE_THRESHOLD:
                dec = f"min{m}: ECE={ece_raw:.5f} < {ECE_THRESHOLD} → calibração não é o problema. " \
                      "Investigar outra causa (features, distribuição, leakage)."
            elif ece_imp >= ECE_IMPROVE_MIN:
                dec = f"min{m}: ECE {ece_raw:.5f} → {ece_rcal:.5f} (melhora {ece_imp:.0%} ≥ 30%) " \
                      "→ ✅ ADOTAR recalibração isotônica."
            else:
                dec = f"min{m}: ECE {ece_raw:.5f} → {ece_rcal:.5f} (melhora {ece_imp:.0%} < 30%) " \
                      "→ ⚠ Calibração marginal. Considerar trocar distribuição (Skellam/NegBinom)."

            decisions.append(dec)
            print(f"  {dec}")

    print("""
─────────────────────────────────────────────────────────────────────────────
SNIPPET DE INTEGRAÇÃO: betsapi_corners_analysis.py, linhas ~2726-2733

  # Carrega calibradores (uma vez, fora do loop)
  import joblib as _jl
  _ISO_CAL_PATH = DATA_DIR / "isotonic_calibrators_ngb.joblib"
  _ngb_iso_cals = _jl.load(_ISO_CAL_PATH) if _ISO_CAL_PATH.exists() else {}

  # Dentro do loop por snap_min, SUBSTITUI o bloco NGBLogNorm:
  if _ngb_model is not None and _ngb_mu_test is not None:
      _ngb_s_te  = _ngb_dist_test.params["s"]
      _ngb_sc_te = _ngb_dist_test.params["scale"]
      # Usa CDF LogNorm nativa (mais preciso que Poisson)
      p_ngb_test = np.array([
          1.0 - sp_lognorm.cdf(
              max(float(np.floor(fl)) + 0.5, 0.01),
              s=float(_ngb_s_te[i]),
              scale=float(_ngb_sc_te[i]))
          for i, fl in enumerate(fl_test)
      ])
      # Aplica calibração isotônica (se disponível para este minuto)
      _iso_entry = _ngb_iso_cals.get(snap_min, {})
      _iso = _iso_entry.get("iso_lognorm")
      if _iso is not None:
          p_ngb_test = np.clip(_iso.predict(p_ngb_test), 1e-6, 1 - 1e-6)
─────────────────────────────────────────────────────────────────────────────
""")


# Classe TE local para não depender de audit_team_encoding.py no walk-forward
class _TargetEncoderSmoothed_local:
    def __init__(self, cols, target_col, smoothing=10):
        self.cols = cols; self.target_col = target_col; self.smoothing = smoothing
        self.encodings_: dict = {}; self.counts_: dict = {}; self.global_mean_: float = 0.0
    def fit(self, df):
        self.global_mean_ = float(df[self.target_col].mean())
        for col in self.cols:
            if col not in df.columns: continue
            s = df.groupby(col)[self.target_col].agg(["mean", "count"])
            sm = s["count"] / (s["count"] + self.smoothing)
            self.encodings_[col] = (sm * s["mean"] + (1 - sm) * self.global_mean_).to_dict()
            self.counts_[col] = s["count"].to_dict()
        return self
    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            m = self.encodings_.get(col, {}); c = self.counts_.get(col, {})
            df[f"{col}_target_enc"] = df[col].map(m).fillna(self.global_mean_)
            rc = df[col].map(c).fillna(0.0)
            df[f"{col}_enc_reliability"] = (rc / (rc + self.smoothing)).round(4)
        return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Recalibração isotônica NGBoost")
    parser.add_argument("--parquet",       default=str(DEFAULT_PARQUET))
    parser.add_argument("--models-dir",    default=str(DEFAULT_MODELS))
    parser.add_argument("--cal-out",       default=str(DEFAULT_CAL_OUT))
    parser.add_argument("--walk-forward",  action="store_true",
                        help="Roda walk-forward com NGBoost retreinado por fold")
    parser.add_argument("--n-estimators",  type=int, default=200,
                        help="n_estimators para o walk-forward NGBoost (padrão: 200)")
    parser.add_argument("--minutes",       type=int, nargs="+", default=None)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Pula Parte 1 (calibração no split fixo)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    models_dir   = Path(args.models_dir)
    cal_out_path = Path(args.cal_out)

    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}")
        sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df_all = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df_all):,} linhas × {df_all.shape[1]} colunas")

    minutes = args.minutes or SNAPSHOT_MINUTES

    # Carrega metadata
    try:
        import joblib
        meta_path = models_dir / "modelo_corners_meta.joblib"
        meta = joblib.load(meta_path) if meta_path.exists() else {}
    except Exception:
        meta = {}

    # ── Parte 1: Calibração no split fixo ──
    report_df = pd.DataFrame()
    if not args.skip_calibration:
        print("\n" + "═" * 70)
        print("  PARTE 1 — CALIBRAÇÃO NO SPLIT FIXO (60/20/20)")
        print("=" * 70)
        result = run_calibration(df_all, models_dir, cal_out_path, meta, minutes)
        report_df = result.get("report", pd.DataFrame())

        if not report_df.empty:
            out_csv = models_dir / "recalibration_report.csv"
            report_df.to_csv(out_csv, index=False)
            print(f"  Relatório salvo: {out_csv}")

    # ── Parte 2: Walk-forward ──
    wf_df = None
    if args.walk_forward:
        print("\n" + "═" * 70)
        print("  PARTE 2 — WALK-FORWARD  (NGBoost retreinado por fold)")
        print(f"  n_estimators={args.n_estimators}")
        print("=" * 70)
        wf_df = run_walk_forward(df_all, models_dir, meta, minutes, args.n_estimators)
        if not wf_df.empty:
            wf_csv = models_dir / "wf_calibration_comparison.csv"
            wf_df.to_csv(wf_csv, index=False)
            print(f"  Walk-forward salvo: {wf_csv}")

    # ── Parte 3: Decisão ──
    print_decision(report_df, wf_df)


if __name__ == "__main__":
    main()
