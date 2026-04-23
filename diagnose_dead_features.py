"""
diagnose_dead_features.py
=========================
Diagnóstico de features mortas no dataset de escanteios.

Uso:
    python diagnose_dead_features.py
    python diagnose_dead_features.py --sanity          # compara MAE antes/depois
    python diagnose_dead_features.py --parquet caminho/para/features.parquet
    python diagnose_dead_features.py --models-dir dados_escanteios
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_PARQUET = Path("dados_escanteios/features_ml.parquet")
DEFAULT_MODELS_DIR = Path("dados_escanteios")
SNAPSHOT_MINUTES = [15, 30, 45, 60, 75]
TARGET = "target_corners_total"

# Lista completa das features suspeitas (SHAP=0 em todos os minutos)
SUSPICIOUS_FEATURES: list[str] = [
    # Aceleração
    "acceleration_attacks",
    "acceleration_corners",
    "acceleration_dangerous",
    # Janelas temporais
    "attacks_last_10min",
    "dangerous_attacks_last_5min",
    "dangerous_attacks_last_10min",
    "attacks_last_5min",
    # Stats de ataques brutas
    "attacks_home",
    "attacks_away",
    "attacks_diff",
    # Shots
    "shots_on_target_home",
    "shots_on_target_away",
    "shots_per_corner",
    # Dinâmica de escanteios
    "corners_acceleration_5_10",
    "corner_jerk",
    "corner_burstiness",
    # Momentum
    "momentum_score",
    "momentum_shift",
    "pace_acceleration",
    # Pressão situacional
    "losing_team_pressure",
    "losing_x_acceleration",
    "comeback_pressure",
    "losing_team_pressure_ratio",
    # Projeções
    "projected_corners_rate5",
    "projected_corners_rate10",
    "projected_da_remaining",
    # Rolling pressure
    "rolling_pressure_5",
    "rolling_pressure_10",
    "time_decay_pressure",
    # Urgência / pressão derivada
    "urgency_factor",
    "urgency_weighted",
    "urgency",
    "urgency_index",
    "corner_drought_pressure",
    "wasted_pressure",
    "unconverted_pressure",
    # Intensidade e regime
    "activity_spike",
    "intensity_drop",
    "intensity_index",
    "pressure_acceleration",
    "da_pressure_acceleration",
    # Eficiência e conversão
    "recent_conversion_efficiency",
    "corner_conversion",
    "shots_per_dangerous_attack",
    "corners_per_dangerous_attack",
    "shots_to_dangerous_ratio",
    "dangerous_to_attacks_ratio",
    "conversion_drop",
    # Expected / projetado (5 min)
    "expected_corners_5min",
    "corners_vs_expected_5",
    # Flags de pico
    "corner_burst_flag",
    "sustained_pressure_flag",
    # Slowdown / dominância
    "winning_team_slowdown",
    "dominance_abs",
    # Pressão de perda tardia
    "losing_in_last_15",
    "final_pressure",
    "late_game_pressure",
    # Construção de pressão sem escanteio
    "building_pressure_no_corner",
    # Ritmo vs global
    "att_rate_short_vs_total",
    "da_rate_short_vs_total",
    "att_acceleration_vs_global",
    "da_acceleration_vs_global",
    # Outros suspeitos comuns
    "pressure_index_5",
    "corners_per_minute_recent",
    "corners_to_dangerous_ratio",
    "time_since_last_corner",
    "time_since_last_shot",
    "time_since_last_dangerous_attack",
]

# Limiares de diagnóstico
THRESH_STD_CONST = 1e-9        # std abaixo disso → CONSTANTE
THRESH_STD_QUASI = 0.001       # std abaixo disso → QUASE CONSTANTE
THRESH_NUNIQUE_QUASI = 10      # poucos valores únicos → QUASE CONSTANTE
THRESH_NAN = 0.5               # >50% NaN → MAJORITARIAMENTE NaN
THRESH_ZERO = 0.95             # >95% zero → MAJORITARIAMENTE ZERO


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _pct_zero_safe(s: pd.Series) -> float:
    """Fraction of zeros, ignoring NaN."""
    valid = s.dropna()
    if len(valid) == 0:
        return float("nan")
    return float((valid == 0).mean())


def classify(nuniq: int, std: float, pct_nan: float, pct_zero: float) -> str:
    if std <= THRESH_STD_CONST or nuniq <= 1:
        return "CONSTANTE"
    if pct_nan > THRESH_NAN:
        return "MAJORITARIAMENTE NaN"
    if std < THRESH_STD_QUASI or nuniq < THRESH_NUNIQUE_QUASI:
        return "QUASE CONSTANTE"
    if pct_zero > THRESH_ZERO:
        return "MAJORITARIAMENTE ZERO"
    return "SAUDÁVEL"


def std_by_minute(df: pd.DataFrame, col: str) -> dict[int, float]:
    result: dict[int, float] = {}
    if "snap_minute" not in df.columns:
        return {m: float("nan") for m in SNAPSHOT_MINUTES}
    for m in SNAPSHOT_MINUTES:
        sub = df.loc[df["snap_minute"] == m, col]
        result[m] = float(sub.std()) if len(sub) > 1 else float("nan")
    return result


# ---------------------------------------------------------------------------
# PART 1 — DIAGNÓSTICO
# ---------------------------------------------------------------------------
def run_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    # Só analisa features que existem no parquet
    features_present = [f for f in SUSPICIOUS_FEATURES if f in df.columns]
    features_missing = [f for f in SUSPICIOUS_FEATURES if f not in df.columns]

    if features_missing:
        print(f"\n[AVISO] {len(features_missing)} features da lista NÃO estão no parquet:")
        for f in features_missing:
            print(f"  - {f}")

    print(f"\nAnalisando {len(features_present)} features presentes...\n")

    rows = []
    for i, col in enumerate(features_present, 1):
        s = df[col]
        nuniq = int(s.nunique())
        pct_nan = float(s.isna().mean())
        pct_zero = _pct_zero_safe(s)
        std = float(s.std())
        mn = float(s.min()) if not s.isna().all() else float("nan")
        mx = float(s.max()) if not s.isna().all() else float("nan")
        mean = float(s.mean()) if not s.isna().all() else float("nan")
        diag = classify(nuniq, std, pct_nan, pct_zero if not np.isnan(pct_zero) else 0.0)
        std_per_min = std_by_minute(df, col)

        rows.append({
            "feature": col,
            "diagnostic": diag,
            "nunique": nuniq,
            "pct_nan": round(pct_nan, 4),
            "pct_zero": round(pct_zero, 4) if not np.isnan(pct_zero) else float("nan"),
            "std": round(std, 6),
            "mean": round(mean, 4) if not np.isnan(mean) else float("nan"),
            "min": round(mn, 4) if not np.isnan(mn) else float("nan"),
            "max": round(mx, 4) if not np.isnan(mx) else float("nan"),
            **{f"std_min{m}": round(std_per_min[m], 6) if not np.isnan(std_per_min[m]) else float("nan")
               for m in SNAPSHOT_MINUTES},
        })

        # Progresso simples a cada 10
        if i % 10 == 0 or i == len(features_present):
            print(f"  [{i:3d}/{len(features_present)}] último: {col} → {diag}")

    report = pd.DataFrame(rows)

    # Ordena: primeiro os mortos
    order = ["CONSTANTE", "MAJORITARIAMENTE NaN", "QUASE CONSTANTE",
             "MAJORITARIAMENTE ZERO", "SAUDÁVEL"]
    report["_order"] = report["diagnostic"].map({v: i for i, v in enumerate(order)})
    report = report.sort_values(["_order", "feature"]).drop(columns="_order").reset_index(drop=True)

    return report


def print_report(report: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("  RELATÓRIO DE FEATURES SUSPEITAS")
    print("=" * 90)

    display_cols = ["feature", "diagnostic", "nunique", "pct_nan", "pct_zero",
                    "std", "std_min15", "std_min30", "std_min45", "std_min60", "std_min75"]
    available_display = [c for c in display_cols if c in report.columns]

    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 20,
        "display.width", 160,
        "display.float_format", "{:.6f}".format,
    ):
        print(report[available_display].to_string(index=True))

    print("\n" + "-" * 90)
    print("  RESUMO POR DIAGNÓSTICO")
    print("-" * 90)
    summary = report["diagnostic"].value_counts()
    for diag, cnt in summary.items():
        print(f"  {diag:<30} {cnt:>4} features")
    print()


def print_value_counts(report: pd.DataFrame, df: pd.DataFrame) -> None:
    targets = report[report["diagnostic"].isin(["CONSTANTE", "QUASE CONSTANTE"])]["feature"].tolist()
    if not targets:
        return
    print("\n" + "=" * 90)
    print("  TOP-5 VALORES MAIS FREQUENTES (CONSTANTE / QUASE CONSTANTE)")
    print("=" * 90)
    for col in targets:
        vc = df[col].value_counts(dropna=False).head(5)
        print(f"\n  [{col}]")
        for val, cnt in vc.items():
            print(f"    {str(val):<20}  {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")


# ---------------------------------------------------------------------------
# PART 2 — AÇÕES
# ---------------------------------------------------------------------------
def print_actions(report: pd.DataFrame) -> None:
    remove_diags = {"CONSTANTE", "MAJORITARIAMENTE NaN", "MAJORITARIAMENTE ZERO"}
    investigate_diags = {"QUASE CONSTANTE"}

    to_remove = report[report["diagnostic"].isin(remove_diags)]["feature"].tolist()
    to_investigate = report[report["diagnostic"].isin(investigate_diags)]["feature"].tolist()
    healthy = report[report["diagnostic"] == "SAUDÁVEL"]["feature"].tolist()

    print("\n" + "=" * 90)
    print("  SEÇÃO DE AÇÕES")
    print("=" * 90)

    print("\n── 1. FEATURES PARA REMOVER IMEDIATAMENTE ──────────────────────────────────────────")
    print("   (CONSTANTE + MAJORITARIAMENTE NaN + MAJORITARIAMENTE ZERO)")
    if to_remove:
        print("\nDEAD_FEATURES_REMOVE = [")
        for f in to_remove:
            print(f'    "{f}",')
        print("]")
    else:
        print("  Nenhuma neste grupo.")

    print("\n── 2. FEATURES PARA INVESTIGAR NO CÓDIGO (QUASE CONSTANTE) ─────────────────────────")
    if to_investigate:
        print("\n   Provavelmente bug de cálculo: divisão por zero, fillna(0) silencioso,")
        print("   ou janela temporal calculada errada. Verificar em betsapi_corners_analysis.py\n")
        print("DEAD_FEATURES_INVESTIGATE = [")
        for f in to_investigate:
            print(f'    "{f}",')
        print("]")
    else:
        print("  Nenhuma neste grupo.")

    if healthy:
        print(f"\n── 3. FEATURES 'SAUDÁVEIS' (SHAP=0 é ruído real do XGBoost) ─────────────────────")
        print(f"   {len(healthy)} features abaixo:")
        print("HEALTHY_BUT_LOW_SHAP = [")
        for f in healthy:
            print(f'    "{f}",')
        print("]")

    all_dead = to_remove + to_investigate
    if all_dead:
        print("\n── 4. SNIPPET PARA betsapi_corners_analysis.py ─────────────────────────────────────")
        print("""
   Adicionar ANTES da linha `available = available_train` (≈ linha 2228):

   # ──────────────────────────────────────────────────────────────────
   # Dead features identificadas por diagnose_dead_features.py
   # ──────────────────────────────────────────────────────────────────
   DEAD_FEATURES_REMOVE = [""")
        for f in all_dead:
            print(f'       "{f}",')
        print("""   ]
   available_train = [c for c in available_train if c not in DEAD_FEATURES_REMOVE]
   # ──────────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# PART 3 — SANITY TEST (MAE comparison)
# ---------------------------------------------------------------------------
def run_sanity(df: pd.DataFrame, report: pd.DataFrame, models_dir: Path) -> None:
    try:
        import joblib
        from sklearn.metrics import mean_absolute_error
        from xgboost import XGBRegressor
    except ImportError as e:
        print(f"\n[SANITY] Dependência não encontrada: {e}. Pulando sanity test.")
        return

    dead_features = report[report["diagnostic"] != "SAUDÁVEL"]["feature"].tolist()
    if not dead_features:
        print("\n[SANITY] Nenhuma feature morta encontrada. Nada a remover.")
        return

    print("\n" + "=" * 90)
    print("  SANITY TEST: MAE COM vs SEM FEATURES MORTAS")
    print("=" * 90)
    print(f"  Features a remover: {len(dead_features)}")
    print(f"  Target: {TARGET}\n")

    results = []

    for snap_min in SNAPSHOT_MINUTES:
        df_min = df[df["snap_minute"] == snap_min].copy()
        if len(df_min) < 200:
            print(f"  [min{snap_min}] ⚠ Amostras insuficientes ({len(df_min)}), pulando.")
            continue

        # Ordena temporalmente
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
        elif "event_id" in df_min.columns:
            df_min = df_min.sort_values("event_id").reset_index(drop=True)

        # Remove linhas sem target
        df_min = df_min.dropna(subset=[TARGET])
        n = len(df_min)

        n_test = int(n * 0.20)
        n_cal = int((n - n_test) * 0.25)
        n_train = n - n_test - n_cal

        df_train = df_min.iloc[:n_train].copy()
        df_test = df_min.iloc[n_train + n_cal:].copy()

        # Determina features usadas pelo modelo salvo (se existir)
        model_path = models_dir / f"modelo_corners_xgb_min{snap_min}.joblib"
        mae_original = float("nan")

        if model_path.exists():
            try:
                model_orig = joblib.load(model_path)
                orig_features = model_orig.get_booster().feature_names
                if orig_features:
                    feat_avail = [f for f in orig_features if f in df_test.columns]
                    X_test_orig = df_test[feat_avail].fillna(0).replace(
                        [np.inf, -np.inf], 0
                    )
                    y_test = df_test[TARGET]
                    preds = model_orig.predict(X_test_orig)
                    mae_original = float(mean_absolute_error(y_test, preds))
            except Exception as exc:
                print(f"  [min{snap_min}] Erro ao carregar modelo original: {exc}")
        else:
            print(f"  [min{snap_min}] Modelo original não encontrado em {model_path}")

        # Determina features para retreino
        exclude = set(dead_features) | {TARGET}
        all_numeric = df_train.select_dtypes(include=[np.number]).columns.tolist()
        # Remove targets, IDs e colunas auxiliares
        skip_prefixes = ("target_", "snap_minute", "event_id", "kickoff")
        feat_retrain = [
            c for c in all_numeric
            if not any(c.startswith(p) for p in skip_prefixes)
            and c not in exclude
        ]

        # Filtra features com >99% NaN
        null_pcts = df_train[feat_retrain].isnull().mean()
        feat_retrain = [c for c in feat_retrain if null_pcts[c] < 0.99]

        if len(feat_retrain) < 10:
            print(f"  [min{snap_min}] ⚠ Features insuficientes para retreino.")
            continue

        # Retreino leve (n_estimators=300 para velocidade)
        X_train_r = df_train[feat_retrain].fillna(0).replace([np.inf, -np.inf], 0)
        y_train_r = df_train[TARGET]
        X_test_r  = df_test[feat_retrain].fillna(0).replace([np.inf, -np.inf], 0)
        y_test_r  = df_test[TARGET]

        print(f"  [min{snap_min}] Retreinando com {len(feat_retrain)} features "
              f"({n_train:,} train / {len(df_test):,} test)...")

        model_new = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        model_new.fit(X_train_r, y_train_r, verbose=False)
        preds_new = model_new.predict(X_test_r)
        mae_new = float(mean_absolute_error(y_test_r, preds_new))

        delta = mae_new - mae_original
        verdict = "✅ SAFE" if (np.isnan(delta) or delta <= 0.05) else "⚠ PIORA"

        results.append({
            "snap_min": snap_min,
            "mae_original": mae_original,
            "mae_sem_mortas": mae_new,
            "delta": delta,
            "verdict": verdict,
            "n_feat_orig": len(model_orig.get_booster().feature_names) if model_path.exists() and not np.isnan(mae_original) else "?",
            "n_feat_new": len(feat_retrain),
        })

    if not results:
        print("  Nenhum resultado gerado.")
        return

    res_df = pd.DataFrame(results)
    print("\n" + "-" * 70)
    print("  COMPARAÇÃO DE MAE")
    print("-" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(res_df[["snap_min", "mae_original", "mae_sem_mortas", "delta",
                       "n_feat_orig", "n_feat_new", "verdict"]].to_string(index=False))

    avg_delta = res_df["delta"].mean()
    if avg_delta <= 0.05:
        print(f"\n  ✅ REMOÇÃO SEGURA (delta MAE médio = {avg_delta:+.4f} ≤ +0.05)")
    else:
        print(f"\n  ⚠ CUIDADO: delta MAE médio = {avg_delta:+.4f} > +0.05. Revisar antes de remover.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Diagnóstico de features mortas")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET),
                        help="Caminho para features_ml.parquet")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                        help="Diretório com modelos .joblib")
    parser.add_argument("--sanity", action="store_true",
                        help="Executa comparação de MAE (requer xgboost + joblib)")
    parser.add_argument("--output", default="dados_escanteios/dead_features_report.csv",
                        help="Arquivo CSV de saída")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERRO] Parquet não encontrado: {parquet_path}")
        sys.exit(1)

    print(f"Carregando {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df):,} linhas × {df.shape[1]} colunas")

    if "snap_minute" in df.columns:
        dist = df["snap_minute"].value_counts().sort_index()
        print(f"snap_minute: {dict(dist)}")

    # --- Part 1: Diagnóstico ---
    report = run_diagnostics(df)
    print_report(report)
    print_value_counts(report, df)

    # --- Salva CSV ---
    csv_cols = ["feature", "diagnostic", "nunique", "pct_nan", "pct_zero", "std",
                "std_min15", "std_min30", "std_min45", "std_min60", "std_min75"]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    available_csv = [c for c in csv_cols if c in report.columns]
    report[available_csv].to_csv(out_path, index=False)
    print(f"\n  Relatório salvo em: {out_path}")

    # --- Part 2: Ações ---
    print_actions(report)

    # --- Part 3: Sanity test ---
    if args.sanity:
        run_sanity(df, report, Path(args.models_dir))
    else:
        print("\n  (Para comparar MAE com/sem features mortas, rode com --sanity)\n")


if __name__ == "__main__":
    main()
