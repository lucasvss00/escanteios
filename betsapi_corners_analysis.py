"""
=============================================================================
BetsAPI – Análise Exploratória e Preparação de Features para ML
=============================================================================
Execute APÓS coletar dados com betsapi_corners_collector.py

Uso:
    jupyter notebook betsapi_corners_analysis.py
    # ou:
    python betsapi_corners_analysis.py

Saídas:
    dados_escanteios/features_ml.parquet  ← dataset pronto para treinar modelo
    dados_escanteios/features_ml.csv
=============================================================================
"""

# %%
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("dados_escanteios")

# %%
# =============================================================================
# 1. CARREGAMENTO
# =============================================================================
print("Carregando dados...")
df_snap = pd.read_parquet(DATA_DIR / "snapshots_por_minuto.parquet")
df_pano = pd.read_parquet(DATA_DIR / "panorama_jogos.parquet")

print(f"Snapshots por minuto : {len(df_snap):,} linhas | {df_snap['event_id'].nunique():,} jogos")
print(f"Panorama (finalizados): {len(df_pano):,} jogos")

# %%
# =============================================================================
# 2. EXPLORAÇÃO BÁSICA
# =============================================================================
print("\n--- Panorama: estatísticas descritivas ---")
cols_interesse = [
    "corners_total", "corners_home_total", "corners_away_total",
    "attacks_home_total", "attacks_away_total",
    "dangerous_attacks_home_total", "dangerous_attacks_away_total",
    "shots_on_target_home_total", "shots_on_target_away_total",
    "total_goals",
]
print(df_pano[cols_interesse].describe().round(2).to_string())

# %%
print("\n--- Distribuição de escanteios por jogo ---")
print(df_pano["corners_total"].value_counts().sort_index().head(20))

# %%
# =============================================================================
# 3. FEATURES PARA ML AO VIVO
#
# Para previsão ao vivo, a lógica é:
#   - Dado o estado do jogo no minuto X, prever:
#     (a) total de escanteios ao final
#     (b) se haverá mais N escanteios nos próximos M minutos
#
# Criamos um dataset onde cada linha é um "snapshot ao vivo" em um minuto
# específico, com features cumulativas até aquele minuto.
# =============================================================================

# Minutos de interesse para previsão ao vivo
SNAPSHOT_MINUTES = [15, 30, 45, 60, 75]

def build_live_features(df_snap: pd.DataFrame, df_pano: pd.DataFrame,
                         snapshot_minutes: list[int]) -> pd.DataFrame:
    """
    Para cada jogo e cada minuto de snapshot:
      - Features: tudo que aconteceu ATÉ aquele minuto
      - Target: total de escanteios do jogo inteiro (de df_pano)
    """
    records = []

    for event_id, group in df_snap.groupby("event_id"):
        group = group.sort_values("minute").reset_index(drop=True)
        pano  = df_pano[df_pano["event_id"] == event_id]
        if pano.empty:
            continue
        pano = pano.iloc[0]

        for snap_min in snapshot_minutes:
            until = group[group["minute"] <= snap_min]
            if until.empty:
                continue

            last = until.iloc[-1]   # último snapshot disponível até snap_min
            n_minutes = len(until)

            # --- Features cumulativas até snap_min ---
            feat = {
                "event_id":    event_id,
                "snap_minute": snap_min,
                "league_id":   last.get("league_id", ""),
                "league_name": last.get("league_name", ""),
                "home_team":   last.get("home_team", ""),
                "away_team":   last.get("away_team", ""),
                "kickoff_dt":  last.get("kickoff_dt", ""),

                # Escanteios até agora
                "corners_home_so_far":  last.get("corners_home"),
                "corners_away_so_far":  last.get("corners_away"),
                "corners_total_so_far": (last.get("corners_home") or 0) +
                                         (last.get("corners_away") or 0),

                # Ritmo de escanteios (por minuto até agora)
                "corners_rate_per_min": round(
                    ((last.get("corners_home") or 0) + (last.get("corners_away") or 0))
                    / max(snap_min, 1), 4
                ),

                # Posse de bola média
                "possession_home_avg": _mean_col(until, "possession_home"),
                "possession_away_avg": _mean_col(until, "possession_away"),

                # Ataques acumulados
                "attacks_home":             last.get("attacks_home"),
                "attacks_away":             last.get("attacks_away"),
                "dangerous_attacks_home":   last.get("dangerous_attacks_home"),
                "dangerous_attacks_away":   last.get("dangerous_attacks_away"),

                # Chutes
                "shots_on_target_home":  last.get("shots_on_target_home"),
                "shots_on_target_away":  last.get("shots_on_target_away"),
                "shots_off_target_home": last.get("shots_off_target_home"),
                "shots_off_target_away": last.get("shots_off_target_away"),

                # Cartões e faltas
                "yellow_cards_home": last.get("yellow_cards_home"),
                "yellow_cards_away": last.get("yellow_cards_away"),
                "red_cards_home":    last.get("red_cards_home"),
                "red_cards_away":    last.get("red_cards_away"),
                "fouls_home":        last.get("fouls_home"),
                "fouls_away":        last.get("fouls_away"),

                # Score ao vivo no minuto
                "score_home": last.get("score_home"),
                "score_away": last.get("score_away"),

                # Diferenças home - away (assimetria de pressão)
                "corners_diff":          _diff(last, "corners_home", "corners_away"),
                "attacks_diff":          _diff(last, "attacks_home", "attacks_away"),
                "dangerous_attacks_diff":_diff(last, "dangerous_attacks_home",
                                                     "dangerous_attacks_away"),
                "shots_on_target_diff":  _diff(last, "shots_on_target_home",
                                                     "shots_on_target_away"),

                # Aceleração de escanteios (últimos 15 min vs antes)
                "corners_last_15_home": _last_n_minutes(until, snap_min, 15, "corners_home"),
                "corners_last_15_away": _last_n_minutes(until, snap_min, 15, "corners_away"),

                # Número real de snapshots disponíveis
                "n_snap_minutes": n_minutes,
            }

            # --- Target ---
            feat["target_corners_total"] = pano.get("corners_total")
            feat["target_corners_remaining"] = (
                (pano.get("corners_total") or 0) -
                ((last.get("corners_home") or 0) + (last.get("corners_away") or 0))
            )
            # Target binário: haverá pelo menos mais 1 escanteio?
            feat["target_more_corners"] = int((feat["target_corners_remaining"] or 0) > 0)

            records.append(feat)

    return pd.DataFrame(records)


def _mean_col(df, col):
    vals = df[col].dropna()
    return round(vals.mean(), 2) if len(vals) > 0 else None


def _diff(row, col_a, col_b):
    a = row.get(col_a)
    b = row.get(col_b)
    if a is None or b is None:
        return None
    return a - b


def _last_n_minutes(df, current_min, n, col):
    """Diferença no valor de `col` nos últimos N minutos."""
    since = current_min - n
    past = df[df["minute"] <= since]
    recent = df[df["minute"] <= current_min]
    if past.empty or recent.empty:
        return None
    val_past = past.iloc[-1].get(col)
    val_now  = recent.iloc[-1].get(col)
    if val_past is None or val_now is None:
        return None
    return val_now - val_past


# %%
print("\nConstruindo dataset de features por minuto de snapshot...")
df_features = build_live_features(df_snap, df_pano, SNAPSHOT_MINUTES)
print(f"Features dataset: {len(df_features):,} linhas | {df_features['event_id'].nunique():,} jogos")

# %%
# =============================================================================
# 4. LIMPEZA BÁSICA
# =============================================================================
# Remove jogos sem target (dados incompletos)
df_features = df_features[df_features["target_corners_total"].notna()]
df_features = df_features[df_features["target_corners_total"] >= 0]

# Corrige remainings negativos (erro de dados)
df_features["target_corners_remaining"] = df_features["target_corners_remaining"].clip(lower=0)

print(f"\nApós limpeza: {len(df_features):,} amostras")

# %%
print("\n--- Features: correlação com target_corners_total ---")
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
corr = (
    df_features[num_cols]
    .corr()["target_corners_total"]
    .drop("target_corners_total")
    .sort_values(key=abs, ascending=False)
    .head(15)
)
print(corr.round(3).to_string())

# %%
# =============================================================================
# 5. SALVAR DATASET DE FEATURES
# =============================================================================
out_pq  = DATA_DIR / "features_ml.parquet"
out_csv = DATA_DIR / "features_ml.csv"
df_features.to_parquet(out_pq, index=False)
df_features.to_csv(out_csv, index=False)
print(f"\nDataset de features salvo em:\n  {out_pq}\n  {out_csv}")

# %%
# =============================================================================
# 6. EXEMPLO RÁPIDO DE MODELO (XGBoost)
# =============================================================================
print("\n--- Treinando modelo de exemplo (XGBoost) ---")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    import xgboost as xgb

    FEATURE_COLS = [
        "snap_minute",
        "corners_home_so_far", "corners_away_so_far", "corners_total_so_far",
        "corners_rate_per_min",
        "possession_home_avg", "possession_away_avg",
        "attacks_home", "attacks_away",
        "dangerous_attacks_home", "dangerous_attacks_away",
        "shots_on_target_home", "shots_on_target_away",
        "shots_off_target_home", "shots_off_target_away",
        "yellow_cards_home", "yellow_cards_away",
        "fouls_home", "fouls_away",
        "score_home", "score_away",
        "corners_diff", "attacks_diff", "dangerous_attacks_diff", "shots_on_target_diff",
        "corners_last_15_home", "corners_last_15_away",
    ]

    TARGET = "target_corners_total"

    df_model = df_features[FEATURE_COLS + [TARGET]].dropna()
    X = df_model[FEATURE_COLS]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"  MAE  : {mae:.3f} escanteios")
    print(f"  RMSE : {rmse:.3f} escanteios")

    # Importância de features
    feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\nTop 10 features mais importantes:")
    print(feat_imp.sort_values(ascending=False).head(10).round(4).to_string())

    # Salva modelo
    import joblib
    model_path = DATA_DIR / "modelo_corners_xgb.joblib"
    joblib.dump(model, model_path)
    print(f"\nModelo salvo em: {model_path}")

except ImportError as e:
    print(f"  Pacote não instalado ({e}). Instale: pip install xgboost scikit-learn joblib")

# %%
print("\nAnálise concluída!")
