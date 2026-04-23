"""
baseline_comparison.py
======================
Compara o XGBoost treinado contra baselines triviais de previsão de
escanteios.  Roda em <1 minuto no dataset de ~68k amostras de teste.

Uso:
    python baseline_comparison.py
    python baseline_comparison.py --parquet dados_escanteios/features_ml.parquet
    python baseline_comparison.py --models-dir dados_escanteios
    python baseline_comparison.py --no-betting    # pula análise de ROI
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET  = Path("dados_escanteios/features_ml.parquet")
DEFAULT_MODELS   = Path("dados_escanteios")
SNAPSHOT_MINUTES = [15, 30, 45, 60, 75]
TARGET           = "target_corners_total"
ODDS             = 1.83
BREAKEVEN        = 1.0 / ODDS        # ≈ 0.546
MIN_EDGE         = 0.02              # igual ao pipeline principal

CORNER_BUCKETS = [
    ("0-2",  0,  2),
    ("3-5",  3,  5),
    ("6-8",  6,  8),
    ("9+",   9, 99),
]


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


def _safe(arr, default=0.0):
    """Substitui NaN/Inf por default."""
    a = np.array(arr, dtype=float)
    a = np.where(np.isfinite(a), a, default)
    return a


def _mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def _rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - float(y.mean())) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-9))


def _dline_vec(csf_arr, la_arr, rem):
    lines = []
    for csf, la in zip(csf_arr, la_arr):
        try:
            rate = float(la) / 90.0 if (la is not None and np.isfinite(float(la))
                                         and float(la) > 0) else 0.1
        except (TypeError, ValueError):
            rate = 0.1
        lines.append(np.round((float(csf) + rem * rate) * 2) / 2)
    return np.array(lines, dtype=float)


def _p_over_poisson(line_arr, mu_arr):
    fl = np.floor(line_arr).astype(int)
    return np.array([1.0 - sp_poisson.cdf(int(fl[i]), mu=max(float(mu_arr[i]), 0.01))
                     for i in range(len(mu_arr))])


def _roi_at_thresh(p_over, over_actual, thresh):
    mask   = p_over >= thresh
    n      = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    wins   = int(over_actual[mask].sum())
    profit = wins * (ODDS - 1) - (n - wins)
    return float(profit / n), n


def _best_roi(p_over_cal, over_actual_cal, p_over_te, over_actual_te,
              min_bets=30):
    """Seleciona threshold no CAL e avalia no TEST."""
    best_thresh, best_cal_roi = BREAKEVEN + MIN_EDGE, -999.0
    for thr in np.arange(BREAKEVEN + MIN_EDGE, 0.73, 0.01):
        mask = p_over_cal >= thr
        n    = int(mask.sum())
        if n < min_bets:
            continue
        wins = int(over_actual_cal[mask].sum())
        r    = (wins * (ODDS - 1) - (n - wins)) / n
        if r > best_cal_roi:
            best_cal_roi, best_thresh = r, thr
    roi_te, n_bets = _roi_at_thresh(p_over_te, over_actual_te, best_thresh)
    return roi_te, n_bets, best_thresh


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


# ---------------------------------------------------------------------------
# BASELINES
# ---------------------------------------------------------------------------
def compute_baselines(
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    train_mean: float,
    snap_min: int,
) -> dict[str, np.ndarray]:
    """Retorna dict {nome: predições} para os 6 baselines."""
    n   = len(y_te)
    rem = 90 - snap_min

    # Coluna de corners acumulados
    csf = _safe(df_te["corners_total_so_far"].values
                if "corners_total_so_far" in df_te.columns
                else np.zeros(n))
    csf = np.clip(csf, 0, 90)

    # Liga média (com fallback)
    league_avg = (_safe(df_te["league_avg_corners"].values, default=np.nan)
                  if "league_avg_corners" in df_te.columns
                  else np.full(n, np.nan))
    league_avg_fill = np.where(np.isfinite(league_avg), league_avg, train_mean)

    # Histórico dos times
    hist_home = (_safe(df_te["hist_home_corners_home_avg"].values, default=np.nan)
                 if "hist_home_corners_home_avg" in df_te.columns
                 else np.full(n, np.nan))
    hist_away = (_safe(df_te["hist_away_corners_away_avg"].values, default=np.nan)
                 if "hist_away_corners_away_avg" in df_te.columns
                 else np.full(n, np.nan))
    # Fallback para média da liga quando histórico está ausente
    hist_home_f = np.where(np.isfinite(hist_home), hist_home, league_avg_fill / 2)
    hist_away_f = np.where(np.isfinite(hist_away), hist_away, league_avg_fill / 2)

    # expected_remaining_corners
    exp_rem = (_safe(df_te["expected_remaining_corners"].values, default=np.nan)
               if "expected_remaining_corners" in df_te.columns
               else np.full(n, np.nan))
    exp_rem_f = np.where(np.isfinite(exp_rem), exp_rem,
                          (league_avg_fill / 90.0) * rem)

    # B0: constante global (média do treino)
    b0 = np.full(n, train_mean)

    # B1: extrapolação linear de taxa
    b1 = csf * (90.0 / snap_min) if snap_min > 0 else b0.copy()

    # B2: extrapolação com piso (nunca prevê menos do que já aconteceu)
    b2 = np.maximum(csf, b1)

    # B3: média histórica dos times
    b3 = hist_home_f + hist_away_f

    # B4: corners_so_far + expected_remaining
    b4 = csf + exp_rem_f

    # B5: Poisson ingênuo (liga)
    b5 = csf + (league_avg_fill / 90.0) * rem

    return {
        "B0_constante":   np.clip(b0, 0, 35),
        "B1_extrapolacao": np.clip(b1, 0, 35),
        "B2_extrapiso":   np.clip(b2, 0, 35),
        "B3_historico":   np.clip(b3, 0, 35),
        "B4_esperado":    np.clip(b4, 0, 35),
        "B5_poisson_liga": np.clip(b5, 0, 35),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(args) -> None:
    try:
        import joblib
    except ImportError:
        print("[ERRO] joblib necessário: pip install joblib")
        sys.exit(1)

    parquet_path = Path(args.parquet)
    models_dir   = Path(args.models_dir)

    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}")
        sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df_all = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df_all):,} linhas × {df_all.shape[1]} colunas\n")

    meta_path = models_dir / "modelo_corners_meta.joblib"
    meta = joblib.load(meta_path) if meta_path.exists() else {}

    # Acumula resultados
    summary_rows: list[dict] = []
    bucket_rows:  list[dict] = []
    betting_rows: list[dict] = []

    for snap_min in SNAPSHOT_MINUTES:
        print(f"{'═'*65}")
        print(f"  Minuto {snap_min}")
        print(f"{'═'*65}")

        # ── Split temporal ──
        df_min = df_all[df_all["snap_minute"] == snap_min].copy()
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)

        df_tr, df_ca, df_te = _split_temporal(df_min)
        y_tr = df_tr[TARGET].dropna().values.astype(float)
        y_te = df_te[TARGET].dropna().values.astype(float)

        if len(y_te) < 50:
            print(f"  [AVISO] Amostras de teste insuficientes ({len(y_te)}). Pulando.\n")
            continue

        train_mean = float(y_tr.mean())
        print(f"  Treino: {len(y_tr):,}  Teste: {len(y_te):,}  "
              f"media_treino={train_mean:.3f}")

        # ── Target encoding ──
        te_path = models_dir / f"target_encoder_min{snap_min}.joblib"
        if te_path.exists():
            te = joblib.load(te_path)
            df_tr = te.transform(df_tr)
            df_ca = te.transform(df_ca)
            df_te = te.transform(df_te)

        med_path = models_dir / f"train_medians_min{snap_min}.joblib"
        train_medians = joblib.load(med_path) if med_path.exists() else {}

        # ── Features para XGBoost ──
        feature_list: list[str] = []
        if meta and snap_min in meta.get("models", {}):
            fl = meta["models"][snap_min].get("features") or []
            feature_list = [c for c in fl if c in df_te.columns]
        if not feature_list:
            skip  = ("target_", "snap_minute", "event_id", "kickoff")
            num   = df_tr.select_dtypes(include=[np.number]).columns.tolist()
            feature_list = [c for c in num
                            if not any(c.startswith(p) for p in skip) and c != TARGET]
            null_pcts = df_tr[feature_list].isnull().mean()
            feature_list = [c for c in feature_list if null_pcts[c] < 0.99]

        # ── XGBoost predições ──
        xgb_path = models_dir / f"modelo_corners_xgb_min{snap_min}.joblib"
        y_xgb    = None
        if xgb_path.exists() and feature_list:
            xgb_model   = joblib.load(xgb_path)
            # Tenta usar a mesma lista de features do modelo salvo
            try:
                saved_feats = xgb_model.get_booster().feature_names
                if saved_feats:
                    fl_xgb = [c for c in saved_feats if c in df_te.columns]
                else:
                    fl_xgb = feature_list
            except Exception:
                fl_xgb = feature_list

            X_te_xgb = _prepare_X(df_te, fl_xgb, train_medians, train_mean)

            # Calibrador isotônico (se disponível)
            cal_path = models_dir / f"calibrador_iso_min{snap_min}.joblib"
            calibrator = joblib.load(cal_path) if cal_path.exists() else None

            raw_preds = xgb_model.predict(X_te_xgb)
            if calibrator is not None:
                try:
                    raw_preds = calibrator.predict(raw_preds)
                except Exception:
                    pass
            y_xgb = np.clip(raw_preds, 0, 35)
            print(f"  XGBoost carregado ({fl_xgb.__len__()} features)")
        else:
            print(f"  [AVISO] Modelo XGBoost não encontrado ({xgb_path})")

        # ── Baselines ──
        baselines = compute_baselines(df_te, y_te, train_mean, snap_min)

        # ── Métricas globais ──
        row: dict = {"snap_min": snap_min, "n_test": len(y_te),
                     "train_mean": round(train_mean, 3)}
        for name, yhat in baselines.items():
            yhat_a = _safe(yhat)[:len(y_te)]
            mae    = _mae(y_te, yhat_a)
            rmse   = _rmse(y_te, yhat_a)
            r2_v   = _r2(y_te, yhat_a)
            row[f"mae_{name}"]  = round(mae, 4)
            row[f"rmse_{name}"] = round(rmse, 4)
            row[f"r2_{name}"]   = round(r2_v, 4)

        if y_xgb is not None:
            mae_xgb  = _mae(y_te, y_xgb)
            rmse_xgb = _rmse(y_te, y_xgb)
            r2_xgb   = _r2(y_te, y_xgb)
            row["mae_XGBoost"]  = round(mae_xgb, 4)
            row["rmse_XGBoost"] = round(rmse_xgb, 4)
            row["r2_XGBoost"]   = round(r2_xgb, 4)
            # Ganho % vs B1
            mae_b1 = row["mae_B1_extrapolacao"]
            row["ganho_vs_B1_pct"] = round((mae_b1 - mae_xgb) / max(mae_b1, 1e-9) * 100, 2)
        summary_rows.append(row)

        # ── Print resumo ──
        print(f"\n  {'Modelo':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
        print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
        for name in baselines:
            print(f"  {name:<22} {row[f'mae_{name}']:>7.4f} "
                  f"{row[f'rmse_{name}']:>7.4f} {row[f'r2_{name}']:>7.4f}")
        if y_xgb is not None:
            ganho = row.get("ganho_vs_B1_pct", float("nan"))
            verdict = ("✅ agrega" if ganho > 15 else
                       "⚠ marginal" if ganho > 5 else
                       "❌ pouco valor")
            print(f"  {'XGBoost':<22} {row['mae_XGBoost']:>7.4f} "
                  f"{row['rmse_XGBoost']:>7.4f} {row['r2_XGBoost']:>7.4f}  "
                  f"[ganho vs B1: {ganho:+.1f}%  {verdict}]")

        # ── Análise por bucket de corners atuais ──
        print(f"\n  Análise por bucket (corners_total_so_far):")
        print(f"  {'Bucket':<8} {'N':>6} {'MAE_B1':>9} {'MAE_XGB':>9} "
              f"{'Ganho%':>8}  {'Veredicto'}")
        print(f"  {'─'*8} {'─'*6} {'─'*9} {'─'*9} {'─'*8}  {'─'*20}")

        if "corners_total_so_far" in df_te.columns:
            csf_te = _safe(df_te["corners_total_so_far"].values)[:len(y_te)]
        else:
            csf_te = np.zeros(len(y_te))

        for blabel, blo, bhi in CORNER_BUCKETS:
            mask  = (csf_te >= blo) & (csf_te <= bhi)
            n_bkt = int(mask.sum())
            if n_bkt < 20:
                continue
            y_b1_bkt  = baselines["B1_extrapolacao"][:len(y_te)][mask]
            mae_b1_bkt = _mae(y_te[mask], y_b1_bkt)
            if y_xgb is not None:
                mae_xgb_bkt = _mae(y_te[mask], y_xgb[mask])
                gain_bkt    = (mae_b1_bkt - mae_xgb_bkt) / max(mae_b1_bkt, 1e-9) * 100
                verdict_bkt = ("✅ XGB melhor" if gain_bkt > 5 else
                               "⚠ equivalente" if gain_bkt > -2 else
                               "❌ B1 ganha aqui")
                print(f"  {blabel:<8} {n_bkt:>6,} {mae_b1_bkt:>9.4f} "
                      f"{mae_xgb_bkt:>9.4f} {gain_bkt:>+7.1f}%  {verdict_bkt}")
                bucket_rows.append({
                    "snap_min": snap_min, "bucket": blabel, "n": n_bkt,
                    "mae_B1": round(mae_b1_bkt, 4), "mae_XGB": round(mae_xgb_bkt, 4),
                    "ganho_pct": round(gain_bkt, 2),
                })
            else:
                print(f"  {blabel:<8} {n_bkt:>6,} {mae_b1_bkt:>9.4f} {'  -':>9}")

        # ── Análise de apostas (betting comparison) ──
        if not args.no_betting:
            print(f"\n  Análise de apostas (ROI):")

            # Linha dinâmica para o teste
            la_te  = (df_te["league_avg_corners"].values
                      if "league_avg_corners" in df_te.columns
                      else np.full(len(df_te), np.nan))
            la_ca  = (df_ca["league_avg_corners"].values
                      if "league_avg_corners" in df_ca.columns
                      else np.full(len(df_ca), np.nan))
            la_list_te = [v if np.isfinite(v) else None for v in la_te]
            la_list_ca = [v if np.isfinite(v) else None for v in la_ca]

            csf_ca = (df_ca["corners_total_so_far"].values
                      if "corners_total_so_far" in df_ca.columns
                      else np.zeros(len(df_ca)))

            dline_te = _dline_vec(csf_te[:len(y_te)], la_list_te[:len(y_te)],
                                   90 - snap_min)
            dline_ca = _dline_vec(_safe(csf_ca), la_list_ca, 90 - snap_min)

            over_te = (y_te > dline_te).astype(float)
            over_ca = (df_ca[TARGET].dropna().values.astype(float) > dline_ca).astype(float)

            print(f"  {'Modelo':<25} {'ROI(test)':>10} {'N apostas':>10} "
                  f"{'Threshold':>10}  {'Veredicto'}")
            print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}  {'─'*15}")

            # B1 como modelo de apostas (Poisson)
            y_b1_all = baselines["B1_extrapolacao"][:len(y_te)]
            y_b1_ca  = compute_baselines(df_ca, over_ca, train_mean,
                                          snap_min)["B1_extrapolacao"][:len(over_ca)]
            p_b1_te = _p_over_poisson(dline_te, y_b1_all)
            p_b1_ca = _p_over_poisson(dline_ca[:len(over_ca)],
                                       y_b1_ca[:len(over_ca)])
            roi_b1, nb_b1, thr_b1 = _best_roi(p_b1_ca, over_ca[:len(p_b1_ca)],
                                                p_b1_te, over_te)

            # B5 Poisson liga
            y_b5_all = baselines["B5_poisson_liga"][:len(y_te)]
            y_b5_ca  = compute_baselines(df_ca, over_ca, train_mean,
                                          snap_min)["B5_poisson_liga"][:len(over_ca)]
            p_b5_te = _p_over_poisson(dline_te, y_b5_all)
            p_b5_ca = _p_over_poisson(dline_ca[:len(over_ca)],
                                       y_b5_ca[:len(over_ca)])
            roi_b5, nb_b5, thr_b5 = _best_roi(p_b5_ca, over_ca[:len(p_b5_ca)],
                                                p_b5_te, over_te)

            # XGBoost (Poisson sobre ŷ)
            roi_xgb = float("nan"); nb_xgb = 0; thr_xgb = BREAKEVEN + MIN_EDGE
            if y_xgb is not None:
                y_xgb_ca_preds = None
                if xgb_path.exists() and feature_list:
                    X_ca_xgb = _prepare_X(df_ca, fl_xgb, train_medians, train_mean)
                    raw_ca   = xgb_model.predict(X_ca_xgb)
                    if calibrator is not None:
                        try: raw_ca = calibrator.predict(raw_ca)
                        except Exception: pass
                    y_xgb_ca_preds = np.clip(raw_ca, 0, 35)

                if y_xgb_ca_preds is not None:
                    p_xgb_te = _p_over_poisson(dline_te, y_xgb[:len(dline_te)])
                    p_xgb_ca = _p_over_poisson(dline_ca[:len(over_ca)],
                                                y_xgb_ca_preds[:len(over_ca)])
                    roi_xgb, nb_xgb, thr_xgb = _best_roi(
                        p_xgb_ca, over_ca[:len(p_xgb_ca)],
                        p_xgb_te, over_te)

            for (lbl, roi_v, n_b, thr) in [
                ("B1 extrapol. + Poisson", roi_b1, nb_b1, thr_b1),
                ("B5 liga + Poisson",      roi_b5, nb_b5, thr_b5),
                ("XGBoost + Poisson",      roi_xgb, nb_xgb, thr_xgb),
            ]:
                roi_s = f"{roi_v:+.1%}" if not np.isnan(roi_v) else "   N/A"
                thr_s = f"{thr:.0%}"
                verd  = ""
                if lbl.startswith("XGBoost") and not np.isnan(roi_v) and not np.isnan(roi_b1):
                    diff = roi_v - roi_b1
                    verd = (f"Δ vs B1: {diff:+.1%} {'✅' if diff > 0.01 else '⚠' if diff > -0.02 else '❌'}")
                print(f"  {lbl:<25} {roi_s:>10} {n_b:>10,} {thr_s:>10}  {verd}")

            betting_rows.append({
                "snap_min": snap_min,
                "roi_B1_poisson":  round(roi_b1, 4) if not np.isnan(roi_b1) else None,
                "roi_B5_poisson":  round(roi_b5, 4) if not np.isnan(roi_b5) else None,
                "roi_XGBoost":     round(roi_xgb, 4) if not np.isnan(roi_xgb) else None,
                "n_bets_B1":  nb_b1, "n_bets_B5": nb_b5, "n_bets_XGB": nb_xgb,
            })

        print()

    # ── Tabela MAE comparativa ──
    print("\n" + "=" * 100)
    print("  TABELA COMPARATIVA — MAE por minuto")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_rows)

    mae_cols   = ["snap_min", "n_test"] + [f"mae_{b}" for b in
                  ["B0_constante", "B1_extrapolacao", "B2_extrapiso",
                   "B3_historico", "B4_esperado", "B5_poisson_liga",
                   "XGBoost"]] + ["ganho_vs_B1_pct"]
    avail_cols = [c for c in mae_cols if c in summary_df.columns]
    with pd.option_context("display.width", 180, "display.float_format", "{:.4f}".format):
        print(summary_df[avail_cols].to_string(index=False))

    # ── Veredicto final ──
    print("\n" + "=" * 100)
    print("  VEREDITO: EM QUE MINUTOS O MODELO JUSTIFICA SUA COMPLEXIDADE?")
    print("=" * 100)
    for _, r in summary_df.iterrows():
        m = int(r["snap_min"])
        if "ganho_vs_B1_pct" not in r or pd.isna(r["ganho_vs_B1_pct"]):
            print(f"  min{m}: XGBoost não disponível para comparação.")
            continue
        g = float(r["ganho_vs_B1_pct"])
        mae_xgb = float(r.get("mae_XGBoost", 0))
        mae_b1  = float(r.get("mae_B1_extrapolacao", 0))
        if g > 15:
            verdict = f"✅ JUSTIFICA — ganho {g:.1f}% vs B1 (MAE {mae_b1:.4f}→{mae_xgb:.4f})"
        elif g > 5:
            verdict = (f"⚠ MARGINAL — ganho {g:.1f}% vs B1 (MAE {mae_b1:.4f}→{mae_xgb:.4f}). "
                       "Verificar se ROI walk-forward justifica.")
        else:
            verdict = (f"❌ POUCO VALOR — ganho apenas {g:.1f}% vs B1 (MAE {mae_b1:.4f}→{mae_xgb:.4f}). "
                       "Considerar usar B1 em produção para este minuto.")
        print(f"  min{m}: {verdict}")

    # ── Salva CSVs ──
    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_df.empty:
        p = out_dir / "baseline_comparison.csv"
        summary_df.to_csv(p, index=False)
        print(f"\n  Tabela salva: {p}")

    if bucket_rows:
        p = out_dir / "baseline_bucket_analysis.csv"
        pd.DataFrame(bucket_rows).to_csv(p, index=False)
        print(f"  Bucket analysis salvo: {p}")

    if betting_rows:
        p = out_dir / "baseline_betting_comparison.csv"
        pd.DataFrame(betting_rows).to_csv(p, index=False)
        print(f"  Betting comparison salvo: {p}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Comparação de baselines")
    parser.add_argument("--parquet",     default=str(DEFAULT_PARQUET))
    parser.add_argument("--models-dir",  default=str(DEFAULT_MODELS))
    parser.add_argument("--no-betting",  action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
