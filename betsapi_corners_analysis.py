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
cols_interesse = [c for c in cols_interesse if c in df_pano.columns]
if cols_interesse:
    print(df_pano[cols_interesse].describe().round(2).to_string())

# %%
print("\n--- Distribuição de escanteios por jogo ---")
if "corners_total" in df_pano.columns:
    print(df_pano["corners_total"].value_counts().sort_index().head(20))

# %%
# =============================================================================
# 2.5 HISTÓRICO DOS TIMES (rolling averages dos últimos N jogos)
#
# Para cada jogo no panorama, calcula médias históricas dos times
# com base nos jogos ANTERIORES (sem data leaking).
# =============================================================================

ROLLING_WINDOW = 10  # últimos N jogos

def build_team_history(df_pano: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Para cada jogo no panorama, calcula features históricas de cada time
    (home e away) com base nos últimos `window` jogos finalizados anteriores.

    Retorna DataFrame com event_id + features históricas.
    """
    df = df_pano.copy()
    if "kickoff_dt" not in df.columns:
        return pd.DataFrame({"event_id": df["event_id"]})

    df["kickoff_dt_parsed"] = pd.to_datetime(df["kickoff_dt"], errors="coerce")
    df = df.sort_values("kickoff_dt_parsed").reset_index(drop=True)

    # Colunas de stats que queremos agregar por time
    stat_cols = {
        "corners_total":                "corners",
        "corners_home_total":           "corners_scored",
        "corners_away_total":           "corners_conceded",
        "dangerous_attacks_home_total": "dangerous_attacks",
        "attacks_home_total":           "attacks",
        "shots_on_target_home_total":   "shots_on_target",
        "total_goals":                  "goals",
    }
    # Filtra apenas colunas que existem
    stat_cols = {k: v for k, v in stat_cols.items() if k in df.columns}

    # Acumula histórico por time (home_id e away_id)
    team_history: dict[str, list[dict]] = {}
    last_game_date: dict[str, pd.Timestamp] = {}
    result_rows = []

    for _, row in df.iterrows():
        home_id = str(row.get("home_id", ""))
        away_id = str(row.get("away_id", ""))
        event_id = row["event_id"]
        current_dt = row.get("kickoff_dt_parsed")

        feat = {"event_id": event_id}

        # --- Dias de descanso desde o último jogo ---
        if home_id and home_id in last_game_date and pd.notna(current_dt) and pd.notna(last_game_date[home_id]):
            feat["days_rest_home"] = int((current_dt - last_game_date[home_id]).days)
        else:
            feat["days_rest_home"] = None

        if away_id and away_id in last_game_date and pd.notna(current_dt) and pd.notna(last_game_date[away_id]):
            feat["days_rest_away"] = int((current_dt - last_game_date[away_id]).days)
        else:
            feat["days_rest_away"] = None

        # Features históricas do time da casa
        if home_id and home_id in team_history:
            hist = team_history[home_id][-window:]
            for orig_col, alias in stat_cols.items():
                vals = [h.get(orig_col) for h in hist if h.get(orig_col) is not None]
                feat[f"hist_home_{alias}_avg"] = round(np.mean(vals), 2) if vals else None
            feat["hist_home_games"] = len(hist)
            # Mando: média de escanteios apenas nos jogos em CASA
            home_games = [h for h in hist if h.get("_is_home") is True]
            hh_vals = [h.get("corners_home_total") for h in home_games
                       if h.get("corners_home_total") is not None]
            feat["hist_home_corners_home_avg"] = round(np.mean(hh_vals), 2) if hh_vals else None
        else:
            for alias in stat_cols.values():
                feat[f"hist_home_{alias}_avg"] = None
            feat["hist_home_games"] = 0
            feat["hist_home_corners_home_avg"] = None

        # Features históricas do time visitante
        if away_id and away_id in team_history:
            hist = team_history[away_id][-window:]
            for orig_col, alias in stat_cols.items():
                vals = [h.get(orig_col) for h in hist if h.get(orig_col) is not None]
                feat[f"hist_away_{alias}_avg"] = round(np.mean(vals), 2) if vals else None
            feat["hist_away_games"] = len(hist)
            # Mando: média de escanteios apenas nos jogos FORA
            # (corners_home_total já foi invertido para away_stats → representa corners do time visitante)
            away_games = [h for h in hist if h.get("_is_home") is False]
            aa_vals = [h.get("corners_home_total") for h in away_games
                       if h.get("corners_home_total") is not None]
            feat["hist_away_corners_away_avg"] = round(np.mean(aa_vals), 2) if aa_vals else None
        else:
            for alias in stat_cols.values():
                feat[f"hist_away_{alias}_avg"] = None
            feat["hist_away_games"] = 0
            feat["hist_away_corners_away_avg"] = None

        result_rows.append(feat)

        # Atualiza histórico: time da casa jogou em casa
        game_stats = {col: row.get(col) for col in stat_cols}
        game_stats["_is_home"] = True
        team_history.setdefault(home_id, []).append(game_stats)

        # Para o visitante, inverte home/away nas stats de escanteios e marca como away
        away_stats = dict(game_stats)
        if "corners_home_total" in away_stats and "corners_away_total" in df.columns:
            away_stats["corners_home_total"] = row.get("corners_away_total")
            away_stats["corners_away_total"] = row.get("corners_home_total")
        away_stats["_is_home"] = False
        team_history.setdefault(away_id, []).append(away_stats)

        # Atualiza última data de jogo
        if home_id and pd.notna(current_dt):
            last_game_date[home_id] = current_dt
        if away_id and pd.notna(current_dt):
            last_game_date[away_id] = current_dt

    return pd.DataFrame(result_rows)


def build_league_avg_corners(df_pano: pd.DataFrame) -> dict[str, float | None]:
    """
    Para cada jogo, calcula a média histórica de corners_total da liga (league_id)
    com base exclusivamente nos jogos ANTERIORES (sem data leaking).
    Retorna dict {event_id: avg_corners_total_da_liga}.
    """
    if "league_id" not in df_pano.columns or "corners_total" not in df_pano.columns:
        return {}

    df = df_pano.copy()
    df["kickoff_dt_parsed"] = pd.to_datetime(df.get("kickoff_dt"), errors="coerce")
    df = df.sort_values("kickoff_dt_parsed").reset_index(drop=True)

    result: dict[str, float | None] = {}
    league_history: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        event_id = row["event_id"]
        league_id = str(row.get("league_id", ""))
        ct = row.get("corners_total")

        # Calcula com base nos jogos ANTERIORES desta liga
        if league_id and league_id in league_history:
            vals = league_history[league_id]
            result[event_id] = round(sum(vals) / len(vals), 2)
        else:
            result[event_id] = None

        # Atualiza APÓS calcular (sem data leaking)
        if league_id and ct is not None:
            league_history.setdefault(league_id, []).append(float(ct))

    return result


print("\nCalculando histórico dos times (rolling window=%d)..." % ROLLING_WINDOW)
df_team_hist = build_team_history(df_pano)
print(f"Histórico calculado para {len(df_team_hist):,} jogos")

print("Calculando média histórica de escanteios por liga...")
league_avg = build_league_avg_corners(df_pano)


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
                         df_hist: pd.DataFrame,
                         snapshot_minutes: list[int],
                         league_avg_map: dict | None = None) -> pd.DataFrame:
    """
    Para cada jogo e cada minuto de snapshot:
      - Features: tudo que aconteceu ATÉ aquele minuto
      - Features pré-jogo: odds, H2H, histórico dos times
      - Features temporais: dia da semana, hora, mês
      - Target: total de escanteios do jogo inteiro (de df_pano)
    """
    # Pré-indexa panorama e histórico por event_id
    pano_idx = df_pano.set_index("event_id") if "event_id" in df_pano.columns else pd.DataFrame()
    hist_idx = df_hist.set_index("event_id") if "event_id" in df_hist.columns else pd.DataFrame()

    records = []

    for event_id, group in df_snap.groupby("event_id"):
        if event_id not in pano_idx.index:
            continue
        pano = pano_idx.loc[event_id]
        if isinstance(pano, pd.DataFrame):
            pano = pano.iloc[0]

        hist = hist_idx.loc[event_id] if event_id in hist_idx.index else pd.Series(dtype=object)
        if isinstance(hist, pd.DataFrame):
            hist = hist.iloc[0]

        group = group.sort_values("minute").reset_index(drop=True)

        for snap_min in snapshot_minutes:
            until = group[group["minute"] <= snap_min]
            if until.empty:
                continue

            last = until.iloc[-1]
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

                # Saves, offsides, goal kicks (índices 9-11 do stats_trend)
                "saves_home":        last.get("saves_home"),
                "saves_away":        last.get("saves_away"),
                "offsides_home":     last.get("offsides_home"),
                "offsides_away":     last.get("offsides_away"),
                "goal_kicks_home":   last.get("goal_kicks_home"),
                "goal_kicks_away":   last.get("goal_kicks_away"),

                # Score ao vivo no minuto
                "score_home": last.get("score_home"),
                "score_away": last.get("score_away"),

                # Contexto de placar
                "score_diff": _safe_sub(last.get("score_home"), last.get("score_away")),
                "total_red_cards": (last.get("red_cards_home") or 0) +
                                    (last.get("red_cards_away") or 0),
                "red_card_diff": _safe_sub(last.get("red_cards_home"),
                                           last.get("red_cards_away")),

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

                # Média histórica de escanteios da liga (jogos anteriores, sem data leak)
                "league_avg_corners": league_avg_map.get(str(event_id)) if league_avg_map else None,

                # Taxa de ataques perigosos por minuto
                "dangerous_attacks_rate": round(
                    ((last.get("dangerous_attacks_home") or 0) +
                     (last.get("dangerous_attacks_away") or 0))
                    / max(snap_min, 1), 4
                ),

                # Proporção de escanteios por ataque total
                "corners_per_attack_ratio": round(
                    ((last.get("corners_home") or 0) + (last.get("corners_away") or 0))
                    / max((last.get("attacks_home") or 0) + (last.get("attacks_away") or 0), 1),
                    4
                ),
            }

            # --- Features pré-jogo do panorama ---
            # Odds
            for col in ["corners_line", "corners_over_odds", "corners_under_odds",
                        "asian_corners_line", "asian_corners_home_odds", "asian_corners_away_odds",
                        "odds_home_win", "odds_draw", "odds_away_win",
                        "goals_line", "goals_over_odds", "goals_under_odds",
                        "btts_yes_odds", "btts_no_odds"]:
                feat[col] = pano.get(col)

            # H2H
            for col in ["h2h_total_games", "h2h_avg_corners_total",
                        "h2h_avg_corners_home", "h2h_avg_corners_away",
                        "h2h_avg_goals_total"]:
                feat[col] = pano.get(col)

            # Stats adicionais do event/view
            for col in ["throw_ins_home_total", "throw_ins_away_total",
                        "tackles_home_total", "tackles_away_total"]:
                feat[col] = pano.get(col)

            # --- Features históricas dos times ---
            for col in hist.index:
                if col != "event_id" and col.startswith("hist_"):
                    feat[col] = hist.get(col)

            # --- Features temporais ---
            kickoff = pd.to_datetime(feat.get("kickoff_dt"), errors="coerce")
            if pd.notna(kickoff):
                feat["day_of_week"]  = kickoff.dayofweek    # 0=segunda, 6=domingo
                feat["hour_of_day"]  = kickoff.hour
                feat["month"]        = kickoff.month
                feat["is_weekend"]   = int(kickoff.dayofweek >= 5)
            else:
                feat["day_of_week"]  = None
                feat["hour_of_day"]  = None
                feat["month"]        = None
                feat["is_weekend"]   = None

            # --- Target ---
            feat["target_corners_total"] = pano.get("corners_total")
            feat["target_corners_remaining"] = (
                (pano.get("corners_total") or 0) -
                ((last.get("corners_home") or 0) + (last.get("corners_away") or 0))
            )
            feat["target_more_corners"] = int((feat["target_corners_remaining"] or 0) > 0)

            records.append(feat)

    return pd.DataFrame(records)


def _mean_col(df, col):
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return round(vals.mean(), 2) if len(vals) > 0 else None


def _diff(row, col_a, col_b):
    a = row.get(col_a)
    b = row.get(col_b)
    if a is None or b is None:
        return None
    return a - b


def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def _last_n_minutes(df, current_min, n, col):
    """Diferença no valor de `col` nos últimos N minutos."""
    if col not in df.columns:
        return None
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
df_features = build_live_features(df_snap, df_pano, df_team_hist, SNAPSHOT_MINUTES)
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
    .drop("target_corners_total", errors="ignore")
    .sort_values(key=abs, ascending=False)
    .head(20)
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
        # Tempo do jogo
        "snap_minute",
        # Escanteios ao vivo
        "corners_home_so_far", "corners_away_so_far", "corners_total_so_far",
        "corners_rate_per_min",
        "corners_diff",
        "corners_last_15_home", "corners_last_15_away",
        # Posse
        "possession_home_avg", "possession_away_avg",
        # Ataques
        "attacks_home", "attacks_away",
        "dangerous_attacks_home", "dangerous_attacks_away",
        "attacks_diff", "dangerous_attacks_diff",
        # Chutes
        "shots_on_target_home", "shots_on_target_away",
        "shots_off_target_home", "shots_off_target_away",
        "shots_on_target_diff",
        # Cartões e faltas
        "yellow_cards_home", "yellow_cards_away",
        "red_cards_home", "red_cards_away",
        "fouls_home", "fouls_away",
        # Saves, offsides, goal kicks
        "saves_home", "saves_away",
        "offsides_home", "offsides_away",
        "goal_kicks_home", "goal_kicks_away",
        # Placar e contexto
        "score_home", "score_away",
        "score_diff", "total_red_cards", "red_card_diff",
        # Odds pré-jogo
        "corners_line", "corners_over_odds", "corners_under_odds",
        "asian_corners_line",
        "odds_home_win", "odds_draw", "odds_away_win",
        "goals_line", "goals_over_odds", "goals_under_odds",
        "btts_yes_odds", "btts_no_odds",
        # H2H
        "h2h_total_games", "h2h_avg_corners_total",
        "h2h_avg_corners_home", "h2h_avg_corners_away",
        "h2h_avg_goals_total",
        # Histórico dos times
        "hist_home_corners_avg", "hist_away_corners_avg",
        "hist_home_corners_scored_avg", "hist_away_corners_scored_avg",
        "hist_home_corners_conceded_avg", "hist_away_corners_conceded_avg",
        "hist_home_dangerous_attacks_avg", "hist_away_dangerous_attacks_avg",
        "hist_home_goals_avg", "hist_away_goals_avg",
        "hist_home_games", "hist_away_games",
        # Temporais
        "day_of_week", "hour_of_day", "month", "is_weekend",
    ]

    # Filtra apenas features que existem no dataset (coleta pode não ter todos)
    available_cols = [c for c in FEATURE_COLS if c in df_features.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing_cols:
        print(f"  Aviso: {len(missing_cols)} features ausentes no dataset (serão ignoradas):")
        for c in missing_cols:
            print(f"    - {c}")

    TARGET = "target_corners_total"

    df_model = df_features[available_cols + [TARGET]].dropna()
    print(f"\n  Amostras para treino (após dropna): {len(df_model):,}")
    print(f"  Features utilizadas: {len(available_cols)}")

    if len(df_model) < 50:
        print("  Dados insuficientes para treino. Colete mais jogos.")
    else:
        X = df_model[available_cols]
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
        print(f"\n  MAE  : {mae:.3f} escanteios")
        print(f"  RMSE : {rmse:.3f} escanteios")

        # Importância de features
        feat_imp = pd.Series(model.feature_importances_, index=available_cols)
        print("\nTop 15 features mais importantes:")
        print(feat_imp.sort_values(ascending=False).head(15).round(4).to_string())

        # Salva modelo
        import joblib
        model_path = DATA_DIR / "modelo_corners_xgb.joblib"
        joblib.dump(model, model_path)
        print(f"\nModelo salvo em: {model_path}")

except ImportError as e:
    print(f"  Pacote não instalado ({e}). Instale: pip install xgboost scikit-learn joblib")

# %%
print("\nAnálise concluída!")
