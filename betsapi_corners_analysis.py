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


def build_league_avg_corners(df_pano: pd.DataFrame) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """
    Para cada jogo, calcula a média e o desvio padrão histórico de corners_total
    da liga (league_id) com base exclusivamente nos jogos ANTERIORES (sem data leaking).
    Retorna (dict_avg, dict_std) — cada um {event_id: valor}.
    """
    if "league_id" not in df_pano.columns or "corners_total" not in df_pano.columns:
        return {}, {}

    df = df_pano.copy()
    df["kickoff_dt_parsed"] = pd.to_datetime(df.get("kickoff_dt"), errors="coerce")
    df = df.sort_values("kickoff_dt_parsed").reset_index(drop=True)

    result_avg: dict[str, float | None] = {}
    result_std: dict[str, float | None] = {}
    league_history: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        event_id = row["event_id"]
        league_id = str(row.get("league_id", ""))
        ct = row.get("corners_total")

        # Calcula com base nos jogos ANTERIORES desta liga
        if league_id and league_id in league_history:
            vals = league_history[league_id]
            result_avg[event_id] = round(sum(vals) / len(vals), 2)
            result_std[event_id] = round(float(np.std(vals)), 2) if len(vals) >= 2 else None
        else:
            result_avg[event_id] = None
            result_std[event_id] = None

        # Atualiza APÓS calcular (sem data leaking)
        if league_id and ct is not None:
            league_history.setdefault(league_id, []).append(float(ct))

    return result_avg, result_std


print("\nCalculando histórico dos times (rolling window=%d)..." % ROLLING_WINDOW)
df_team_hist = build_team_history(df_pano)
print(f"Histórico calculado para {len(df_team_hist):,} jogos")

print("Calculando média e desvio padrão histórico de escanteios por liga...")
league_avg, league_std = build_league_avg_corners(df_pano)


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
                         league_avg_map: dict | None = None,
                         league_std_map: dict | None = None) -> pd.DataFrame:
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

            # --- Features granulares adicionais ---
            c_home = feat["corners_home_so_far"] or 0
            c_away = feat["corners_away_so_far"] or 0
            c_total = c_home + c_away

            # Janelas temporais: últimos 5 e 10 minutos
            corners_last_5_home = _last_n_minutes(until, snap_min, 5, "corners_home")
            corners_last_5_away = _last_n_minutes(until, snap_min, 5, "corners_away")
            corners_last_10_home = _last_n_minutes(until, snap_min, 10, "corners_home")
            corners_last_10_away = _last_n_minutes(until, snap_min, 10, "corners_away")

            feat["corners_last_5min"] = ((corners_last_5_home or 0) +
                                          (corners_last_5_away or 0))
            feat["corners_last_10min"] = ((corners_last_10_home or 0) +
                                           (corners_last_10_away or 0))
            feat["corners_acceleration_5_10"] = (feat["corners_last_5min"] -
                                                  feat["corners_last_10min"])
            feat["corners_rate_last_5"] = round(feat["corners_last_5min"] / 5.0, 4)
            feat["corners_rate_last_10"] = round(feat["corners_last_10min"] / 10.0, 4)

            feat["home_corners_last_5"] = corners_last_5_home
            feat["away_corners_last_5"] = corners_last_5_away
            feat["home_corners_last_10"] = corners_last_10_home
            feat["away_corners_last_10"] = corners_last_10_away

            # Estado do jogo
            score_h = last.get("score_home") or 0
            score_a = last.get("score_away") or 0
            feat["is_draw"] = int(score_h == score_a)

            # Pressão do time que lidera/perde (corners últimos 10 min)
            if score_h > score_a:
                feat["leading_team_pressure"] = corners_last_10_home or 0
                feat["losing_team_pressure"] = corners_last_10_away or 0
            elif score_a > score_h:
                feat["leading_team_pressure"] = corners_last_10_away or 0
                feat["losing_team_pressure"] = corners_last_10_home or 0
            else:
                feat["leading_team_pressure"] = 0
                feat["losing_team_pressure"] = 0

            # Tempo restante
            feat["time_remaining"] = 90 - snap_min
            feat["total_time_remaining"] = 90 - snap_min + 3  # ~3 min acréscimo médio

            # Proporção de escanteios por time
            feat["home_corner_share"] = round(c_home / max(c_total, 1), 4)
            feat["away_corner_share"] = round(c_away / max(c_total, 1), 4)

            # Taxa por time
            feat["home_corners_rate"] = round(c_home / max(snap_min, 1), 4)
            feat["away_corners_rate"] = round(c_away / max(snap_min, 1), 4)

            # Expectativa combinada de escanteios (hist home attack + away defense)
            home_avg_for = hist.get("hist_home_corners_scored_avg")
            home_avg_against = hist.get("hist_home_corners_conceded_avg")
            away_avg_for = hist.get("hist_away_corners_scored_avg")
            away_avg_against = hist.get("hist_away_corners_conceded_avg")

            home_exp = None
            away_exp = None
            match_exp = None
            if home_avg_for is not None and away_avg_against is not None:
                home_exp = (float(home_avg_for) + float(away_avg_against)) / 2
            if away_avg_for is not None and home_avg_against is not None:
                away_exp = (float(away_avg_for) + float(home_avg_against)) / 2
            if home_exp is not None and away_exp is not None:
                match_exp = home_exp + away_exp

            feat["home_expected_corners"] = round(home_exp, 4) if home_exp is not None else None
            feat["away_expected_corners"] = round(away_exp, 4) if away_exp is not None else None
            feat["match_expected_corners"] = round(match_exp, 4) if match_exp is not None else None

            # Desvio do ritmo esperado
            if match_exp and match_exp > 0:
                expected_at_min = match_exp * snap_min / 90
                feat["corners_vs_expected"] = round(c_total - expected_at_min, 4)
                feat["pace_ratio"] = round(c_total / max(expected_at_min, 0.1), 4)
            else:
                feat["corners_vs_expected"] = None
                feat["pace_ratio"] = None

            # Liga: desvio padrão e z-score
            _league_avg = league_avg_map.get(str(event_id)) if league_avg_map else None
            _league_std = league_std_map.get(str(event_id)) if league_std_map else None
            feat["league_std_corners"] = _league_std

            if _league_avg is not None and _league_std is not None and _league_std > 0:
                expected_at_min_lg = _league_avg * snap_min / 90
                feat["z_score_corners"] = round(
                    (c_total - expected_at_min_lg) / _league_std, 4)
            else:
                feat["z_score_corners"] = None

            # Estilo do time relativo à liga
            if _league_avg is not None:
                half_league = _league_avg / 2  # média por time na liga
                feat["team_style_home"] = (round(float(home_avg_for) - half_league, 4)
                                            if home_avg_for is not None else None)
                feat["team_style_away"] = (round(float(away_avg_for) - half_league, 4)
                                            if away_avg_for is not None else None)
            else:
                feat["team_style_home"] = None
                feat["team_style_away"] = None

            # Índice de intensidade
            if _league_avg is not None and _league_avg > 0:
                league_rate = _league_avg / 90
                feat["intensity_index"] = round(
                    feat["corners_rate_last_10"] / max(league_rate, 0.01), 4)
            else:
                feat["intensity_index"] = None

            # Late game boost
            feat["late_game_boost"] = round(
                feat["corners_last_10min"] * (snap_min / 90), 4)

            # Índice de pressão por time (últimos 10 min)
            h10 = corners_last_10_home or 0
            a10 = corners_last_10_away or 0
            feat["pressure_index_home"] = round(h10 / max(a10, 0.5), 4)
            feat["pressure_index_away"] = round(a10 / max(h10, 0.5), 4)

            # Fator estado do jogo × tempo
            feat["game_state_factor"] = round((score_h - score_a) * snap_min, 4)

            # Pressão de comeback (time perdendo após min 60)
            feat["comeback_pressure"] = int(snap_min > 60 and score_h != score_a)

            # Dominância de escanteios
            feat["dominance_index"] = round(
                abs(feat["home_corner_share"] - 0.5), 4)

            # Volatilidade (diferença entre ritmos de 5 e 10 min)
            feat["volatility_index"] = round(abs(
                feat["corners_rate_last_5"] - feat["corners_rate_last_10"]), 4)

            # Momentum shift
            feat["momentum_shift"] = round(
                feat["corners_last_5min"] - feat["corners_last_10min"] / 2, 4)

            # Projeção de escanteios restantes
            feat["expected_remaining_corners"] = round(
                feat["corners_rate_per_min"] * feat["total_time_remaining"], 4)
            feat["adjusted_expected_remaining"] = round(
                feat["corners_rate_last_10"] * feat["total_time_remaining"], 4)

            # Análise 1º vs 2º tempo (só para snap_min > 45)
            if snap_min > 45:
                at_45 = group[group["minute"] <= 45]
                if not at_45.empty:
                    last_45 = at_45.iloc[-1]
                    c_45_home = last_45.get("corners_home") or 0
                    c_45_away = last_45.get("corners_away") or 0
                    first_half = c_45_home + c_45_away
                else:
                    first_half = 0

                second_half_so_far = c_total - first_half
                mins_2nd = max(snap_min - 45, 1)
                second_half_rate = second_half_so_far / mins_2nd
                first_half_rate = first_half / 45

                feat["first_half_corners"] = first_half
                feat["second_half_corners_so_far"] = second_half_so_far
                feat["second_half_rate"] = round(second_half_rate, 4)
                feat["delta_rate_halves"] = round(second_half_rate - first_half_rate, 4)
                feat["fatigue_factor"] = (round(
                    feat["corners_rate_last_10"] / max(first_half_rate, 0.01), 4)
                    if first_half > 0 else None)
            else:
                feat["first_half_corners"] = None
                feat["second_half_corners_so_far"] = None
                feat["second_half_rate"] = None
                feat["delta_rate_halves"] = None
                feat["fatigue_factor"] = None

            # Features adicionais sugeridas
            feat["possession_diff"] = _safe_sub(
                feat["possession_home_avg"], feat["possession_away_avg"])
            feat["fouls_diff"] = _diff(last, "fouls_home", "fouls_away")
            feat["fouls_total"] = ((last.get("fouls_home") or 0) +
                                    (last.get("fouls_away") or 0))
            feat["corners_per_dangerous_attack"] = round(
                c_total / max((last.get("dangerous_attacks_home") or 0) +
                               (last.get("dangerous_attacks_away") or 0), 1), 4)
            total_shots = ((last.get("shots_on_target_home") or 0) +
                           (last.get("shots_on_target_away") or 0) +
                           (last.get("shots_off_target_home") or 0) +
                           (last.get("shots_off_target_away") or 0))
            feat["shots_per_corner"] = round(
                total_shots / max(c_total, 1), 4)

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

    # --- Pós-processamento: features de momentum (deltas entre snapshots) ---
    # Para cada jogo, calcula variações entre snapshots consecutivos.
    # Minuto 15 fica sem deltas (sem snapshot anterior).
    MOMENTUM_COLS = [
        "corners_total_so_far", "corners_home_so_far", "corners_away_so_far",
        "corners_rate_per_min", "dangerous_attacks_home", "dangerous_attacks_away",
        "attacks_home", "attacks_away",
        "shots_on_target_home", "shots_on_target_away",
        "possession_home_avg",
    ]

    # Agrupa records por event_id para calcular deltas
    from collections import defaultdict
    event_records: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        event_records[rec["event_id"]].append(i)

    for eid, indices in event_records.items():
        # Ordena por snap_minute
        indices_sorted = sorted(indices, key=lambda i: records[i]["snap_minute"])
        for j in range(len(indices_sorted)):
            idx = indices_sorted[j]
            if j == 0:
                # Primeiro snapshot (min 15): sem deltas
                for col in MOMENTUM_COLS:
                    records[idx][f"delta_{col}"] = None
            else:
                prev_idx = indices_sorted[j - 1]
                for col in MOMENTUM_COLS:
                    curr_val = records[idx].get(col) or 0
                    prev_val = records[prev_idx].get(col) or 0
                    records[idx][f"delta_{col}"] = round(curr_val - prev_val, 4)

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
df_features = build_live_features(df_snap, df_pano, df_team_hist, SNAPSHOT_MINUTES,
                                   league_avg_map=league_avg,
                                   league_std_map=league_std)
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
# =============================================================================
# 4.5 TARGET ENCODING (liga e times)
#
# Substitui IDs categóricos pela média suavizada do target.
# Smoothing Bayesiano: times/ligas com poucos jogos são puxados para a média global.
# =============================================================================

class TargetEncoderSmoothed:
    """Target encoding com smoothing Bayesiano para categorias de alta cardinalidade."""

    def __init__(self, cols: list[str], target_col: str, smoothing: int = 10):
        self.cols = cols
        self.target_col = target_col
        self.smoothing = smoothing
        self.encodings_: dict[str, dict] = {}
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "TargetEncoderSmoothed":
        self.global_mean_ = df[self.target_col].mean()
        for col in self.cols:
            if col not in df.columns:
                continue
            stats = df.groupby(col)[self.target_col].agg(["mean", "count"])
            smooth = stats["count"] / (stats["count"] + self.smoothing)
            stats["encoded"] = smooth * stats["mean"] + (1 - smooth) * self.global_mean_
            self.encodings_[col] = stats["encoded"].to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols:
            new_col = f"{col}_target_enc"
            mapping = self.encodings_.get(col, {})
            df[new_col] = df[col].map(mapping).fillna(self.global_mean_)
        return df


ENCODE_COLS = ["league_id", "home_team", "away_team"]
# Apenas usa colunas que existem no dataset
ENCODE_COLS = [c for c in ENCODE_COLS if c in df_features.columns]

if ENCODE_COLS:
    print("\nAplicando target encoding para:", ENCODE_COLS)
    target_encoder = TargetEncoderSmoothed(
        cols=ENCODE_COLS,
        target_col="target_corners_total",
        smoothing=10,
    )
    target_encoder.fit(df_features)
    df_features = target_encoder.transform(df_features)
    for col in ENCODE_COLS:
        enc_col = f"{col}_target_enc"
        if enc_col in df_features.columns:
            print(f"  {enc_col}: min={df_features[enc_col].min():.2f}  "
                  f"max={df_features[enc_col].max():.2f}  "
                  f"mean={df_features[enc_col].mean():.2f}")

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
# 6. TREINO DE MODELOS — PIPELINE COMPLETO
#
# Melhorias sobre o modelo baseline:
#   1. Modelos separados por minuto de snapshot (15, 30, 45, 60, 75)
#   2. Quantile regression (P10, P50, P90) para intervalos de confiança
#   3. Calibração isotônica para corrigir viés nos extremos
#   4. Features de momentum (delta entre snapshots) — já adicionadas na seção 3
#   5. Target encoding (liga/time) — já adicionado na seção 4.5
# =============================================================================
print("\n" + "═" * 62)
print("  TREINANDO MODELOS AVANÇADOS")
print("═" * 62)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import norm as sp_norm
    import xgboost as xgb
    import joblib

    # --- Features base (usadas por todos os minutos) ---
    BASE_FEATURE_COLS = [
        # Escanteios ao vivo
        "corners_home_so_far", "corners_away_so_far", "corners_total_so_far",
        "corners_rate_per_min",
        "corners_diff",
        "corners_last_15_home", "corners_last_15_away",
        "corners_per_attack_ratio",
        # Janelas temporais: últimos 5 e 10 minutos
        "corners_last_5min", "corners_last_10min",
        "corners_acceleration_5_10",
        "corners_rate_last_5", "corners_rate_last_10",
        "home_corners_last_5", "away_corners_last_5",
        "home_corners_last_10", "away_corners_last_10",
        # Posse
        "possession_home_avg", "possession_away_avg",
        "possession_diff",
        # Ataques
        "attacks_home", "attacks_away",
        "dangerous_attacks_home", "dangerous_attacks_away",
        "attacks_diff", "dangerous_attacks_diff",
        "dangerous_attacks_rate",
        # Chutes
        "shots_on_target_home", "shots_on_target_away",
        "shots_off_target_home", "shots_off_target_away",
        "shots_on_target_diff",
        # Placar e contexto
        "score_home", "score_away",
        "score_diff", "total_red_cards", "red_card_diff",
        "is_draw",
        # Pressão por estado do jogo
        "leading_team_pressure", "losing_team_pressure",
        "game_state_factor", "comeback_pressure",
        # Tempo restante
        "time_remaining", "total_time_remaining",
        # Proporção e taxa por time
        "home_corner_share", "away_corner_share",
        "home_corners_rate", "away_corners_rate",
        # Expectativa combinada (hist home attack + away defense)
        "home_expected_corners", "away_expected_corners", "match_expected_corners",
        "corners_vs_expected", "pace_ratio",
        # Liga
        "league_avg_corners", "league_std_corners",
        "z_score_corners",
        "team_style_home", "team_style_away",
        "intensity_index",
        # Dinâmica do jogo
        "late_game_boost",
        "pressure_index_home", "pressure_index_away",
        "dominance_index", "volatility_index", "momentum_shift",
        # Projeções
        "expected_remaining_corners", "adjusted_expected_remaining",
        # 1º vs 2º tempo
        "first_half_corners", "second_half_corners_so_far",
        "second_half_rate", "delta_rate_halves", "fatigue_factor",
        # Faltas e conversão
        "fouls_diff", "fouls_total",
        "corners_per_dangerous_attack", "shots_per_corner",
        # Histórico dos times
        "hist_home_corners_avg", "hist_away_corners_avg",
        "hist_home_corners_scored_avg", "hist_away_corners_scored_avg",
        "hist_home_corners_conceded_avg", "hist_away_corners_conceded_avg",
        "hist_home_corners_home_avg", "hist_away_corners_away_avg",
        "hist_home_dangerous_attacks_avg", "hist_away_dangerous_attacks_avg",
        "hist_home_goals_avg", "hist_away_goals_avg",
        "hist_home_games", "hist_away_games",
        # Target encoding
        "league_id_target_enc", "home_team_target_enc", "away_team_target_enc",
        # Temporais
        "day_of_week", "hour_of_day", "month", "is_weekend",
    ]

    # Features de momentum (só disponíveis para minutos > 15)
    MOMENTUM_FEATURE_COLS = [
        "delta_corners_total_so_far", "delta_corners_home_so_far",
        "delta_corners_away_so_far", "delta_corners_rate_per_min",
        "delta_dangerous_attacks_home", "delta_dangerous_attacks_away",
        "delta_attacks_home", "delta_attacks_away",
        "delta_shots_on_target_home", "delta_shots_on_target_away",
        "delta_possession_home_avg",
    ]

    TARGET = "target_corners_total"

    def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
        """Filtra features existentes, remove ≥99% NaN, preenche NaN restantes."""
        available = [c for c in feature_cols if c in df.columns]

        # Remove colunas quase vazias
        null_pcts = df[available].isnull().mean()
        available = [c for c in available if null_pcts[c] < 0.99]

        df_out = df[available + [TARGET]].copy()
        df_out = df_out.dropna(subset=[TARGET])

        # Preenche NaN: históricas/liga/encoding → mediana; ao vivo → 0
        fill_med = [c for c in available if c.startswith(("hist_", "league_"))
                    or c.endswith("_target_enc")]
        fill_zero = [c for c in available if c not in fill_med]

        for c in fill_med:
            df_out[c] = df_out[c].fillna(df_out[c].median())
        for c in fill_zero:
            df_out[c] = df_out[c].fillna(0)

        return available, df_out

    # --- Armazenamento de resultados e artefatos ---
    all_metadata: dict = {"snapshot_minutes": SNAPSHOT_MINUTES, "models": {}}

    print(f"\n  Dataset total: {len(df_features):,} amostras")
    print(f"  Split: 60% treino / 20% calibração / 20% teste")

    for snap_min in SNAPSHOT_MINUTES:
        print(f"\n{'─' * 62}")
        print(f"  ⚽ MINUTO {snap_min}")
        print(f"{'─' * 62}")

        df_min = df_features[df_features["snap_minute"] == snap_min].copy()

        # Minuto 15 não tem momentum; demais sim
        if snap_min == 15:
            feat_cols = BASE_FEATURE_COLS
        else:
            feat_cols = BASE_FEATURE_COLS + MOMENTUM_FEATURE_COLS

        available, df_clean = prepare_features(df_min, feat_cols)

        if len(df_clean) < 100:
            print(f"  ⚠ Dados insuficientes ({len(df_clean)} amostras). Pulando.")
            continue

        X = df_clean[available]
        y = df_clean[TARGET]

        # Split 3-way: treino (60%), calibração (20%), teste (20%)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42  # 0.25 × 0.80 = 0.20
        )

        print(f"  Amostras: treino={len(X_train):,}  cal={len(X_cal):,}  teste={len(X_test):,}")
        print(f"  Features: {len(available)}")

        # ==================================================================
        # 6a. Modelo principal (regressão à média — squared error)
        # ==================================================================
        model_mean = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=30,
        )
        model_mean.fit(
            X_train, y_train,
            eval_set=[(X_cal, y_cal)],
            verbose=False,
        )

        # --- Métricas SEM calibração ---
        preds_raw = model_mean.predict(X_test)
        mae_raw  = mean_absolute_error(y_test, preds_raw)
        rmse_raw = mean_squared_error(y_test, preds_raw) ** 0.5

        # ==================================================================
        # 6b. Calibração isotônica (corrige viés nos extremos)
        #
        # Ajusta uma função monotônica não-paramétrica que mapeia
        # predição_raw → predição_corrigida, usando o set de calibração.
        # ==================================================================
        preds_cal_raw = model_mean.predict(X_cal)
        calibrator = IsotonicRegression(y_min=0, y_max=35, out_of_bounds="clip")
        calibrator.fit(preds_cal_raw, y_cal)

        preds_calibrated = calibrator.predict(preds_raw)
        mae_cal  = mean_absolute_error(y_test, preds_calibrated)
        rmse_cal = mean_squared_error(y_test, preds_calibrated) ** 0.5

        # Decide se calibração ajudou ou não
        use_calibration = mae_cal < mae_raw
        preds_best = preds_calibrated if use_calibration else preds_raw
        mae_best = mae_cal if use_calibration else mae_raw

        print(f"\n  Modelo principal (squared error):")
        print(f"    Sem calibração   : MAE={mae_raw:.3f}  RMSE={rmse_raw:.3f}")
        print(f"    Com calibração   : MAE={mae_cal:.3f}  RMSE={rmse_cal:.3f}")
        print(f"    → Usando: {'calibrado ✓' if use_calibration else 'raw (calibração não ajudou)'}")

        # Viés por faixa
        for lo, hi in [(0, 5), (6, 8), (9, 11), (12, 15), (16, 30)]:
            mask = (y_test >= lo) & (y_test <= hi)
            if mask.sum() > 0:
                bias = (preds_best[mask] - y_test[mask].values).mean()
                mae_f = mean_absolute_error(y_test[mask], preds_best[mask])
                print(f"    Faixa {lo:2d}-{hi:2d}: MAE={mae_f:.2f}  viés={bias:+.2f}  (n={mask.sum():,})")

        # ==================================================================
        # 6f. Avaliação da linha dinâmica
        #
        # Linha = corners_acumulados + (minutos_restantes / 10)
        # Arredondada para .0 ou .5 mais próximo.
        # Aposta: sempre Over a linha a odds 1.83.
        # ==================================================================
        remaining_minutes = 90 - snap_min
        if "corners_total_so_far" in X_test.columns:
            corners_so_far_test = X_test["corners_total_so_far"].values
        else:
            corners_so_far_test = np.zeros(len(X_test))

        raw_line = corners_so_far_test + remaining_minutes / 10.0
        dynamic_line = np.round(raw_line * 2) / 2  # arredonda para .0 ou .5

        over_actual = (y_test.values > dynamic_line).astype(int)

        accuracy_dyn = over_actual.mean()
        roi_dyn = accuracy_dyn * 0.83 - (1.0 - accuracy_dyn) * 1.0

        # P(Over) via distribuição normal centrada na predição, σ = RMSE do modelo
        sigma = max(rmse_raw, 0.5)
        p_over = sp_norm.cdf((preds_best - dynamic_line) / sigma)
        brier_dyn = float(np.mean((p_over - over_actual) ** 2))

        print(f"\n  Linha dinâmica  (corners_atual + {remaining_minutes}/10 → arred. .0/.5):")
        print(f"    Linha média           : {dynamic_line.mean():.2f}  "
              f"(min={dynamic_line.min():.1f}  max={dynamic_line.max():.1f})")
        print(f"    Acurácia (over rate)  : {accuracy_dyn:.1%}")
        print(f"    ROI @ odds 1.83       : {roi_dyn:+.1%}")
        print(f"    Brier Score           : {brier_dyn:.4f}")

        # ==================================================================
        # 6c. Quantile regression (P10, P50, P90)
        #
        # NÃO aplica calibração isotônica nos quantis — isso colapsa
        # os intervalos para a média. Os modelos quantile são usados raw.
        # ==================================================================
        quantile_models = {}

        for q_name, q_alpha in [("q10", 0.10), ("q50", 0.50), ("q90", 0.90)]:
            model_q = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q_alpha,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30,
            )
            model_q.fit(
                X_train, y_train,
                eval_set=[(X_cal, y_cal)],
                verbose=False,
            )
            quantile_models[q_name] = model_q

        # Avalia intervalo de confiança no teste (sem calibração)
        p10_test = quantile_models["q10"].predict(X_test)
        p50_test = quantile_models["q50"].predict(X_test)
        p90_test = quantile_models["q90"].predict(X_test)

        coverage = ((y_test.values >= p10_test) & (y_test.values <= p90_test)).mean()
        interval_width = (p90_test - p10_test).mean()
        mae_p50 = mean_absolute_error(y_test, p50_test)

        # Cobertura real por quantil (sanity check)
        below_p10 = (y_test.values < p10_test).mean()
        above_p90 = (y_test.values > p90_test).mean()

        print(f"\n  Quantile regression:")
        print(f"    P50 MAE           : {mae_p50:.3f}")
        print(f"    Intervalo P10-P90 : largura média = {interval_width:.1f} corners")
        print(f"    Cobertura P10-P90 : {coverage:.1%} (ideal: ~80%)")
        print(f"    Abaixo do P10     : {below_p10:.1%} (ideal: ~10%)")
        print(f"    Acima do P90      : {above_p90:.1%} (ideal: ~10%)")

        # ==================================================================
        # 6d. Feature importance (top 10)
        # ==================================================================
        feat_imp = pd.Series(model_mean.feature_importances_, index=available)
        top10 = feat_imp.sort_values(ascending=False).head(10)
        print(f"\n  Top 10 features:")
        for i, (feat, val) in enumerate(top10.items(), 1):
            print(f"    {i:2d}. {feat:<42s} {val:.4f}")

        # ==================================================================
        # 6e. Salva artefatos
        # ==================================================================
        joblib.dump(model_mean,  DATA_DIR / f"modelo_corners_xgb_min{snap_min}.joblib")
        if use_calibration:
            joblib.dump(calibrator,  DATA_DIR / f"calibrador_iso_min{snap_min}.joblib")

        for q_name in ["q10", "q50", "q90"]:
            joblib.dump(quantile_models[q_name],
                        DATA_DIR / f"modelo_corners_xgb_min{snap_min}_{q_name}.joblib")

        all_metadata["models"][snap_min] = {
            "features":         available,
            "n_train":          len(X_train),
            "n_cal":            len(X_cal),
            "n_test":           len(X_test),
            "mae_raw":          round(mae_raw, 4),
            "mae_calibrated":   round(mae_cal, 4),
            "use_calibration":  use_calibration,
            "mae_best":         round(mae_best, 4),
            "rmse_raw":         round(rmse_raw, 4),
            "mae_p50":          round(mae_p50, 4),
            "coverage_80":      round(coverage, 4),
            "interval_width":   round(interval_width, 2),
            "dynamic_line_accuracy": round(accuracy_dyn, 4),
            "dynamic_line_roi":      round(roi_dyn, 4),
            "dynamic_line_brier":    round(brier_dyn, 4),
        }

    # --- Salva target encoder e metadata ---
    if ENCODE_COLS:
        joblib.dump(target_encoder, DATA_DIR / "target_encoder.joblib")
        print(f"\n  Target encoder salvo → {DATA_DIR / 'target_encoder.joblib'}")

    joblib.dump(all_metadata, DATA_DIR / "modelo_corners_meta.joblib")
    print(f"  Metadata salvo → {DATA_DIR / 'modelo_corners_meta.joblib'}")

    # --- Resumo final ---
    print(f"\n{'═' * 62}")
    print(f"  RESUMO DOS MODELOS")
    print(f"{'═' * 62}")
    print(f"  {'Min':>4s}  {'MAE best':>9s}  {'MAE P50':>8s}  {'P10-P90':>8s}  {'Cobert':>7s}  {'Cal?':>4s}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*4}")
    for m, info in all_metadata["models"].items():
        cal_flag = "✓" if info["use_calibration"] else "✗"
        print(f"  {m:>4d}  {info['mae_best']:>9.3f}  {info['mae_p50']:>8.3f}  "
              f"{info['interval_width']:>7.1f}  {info['coverage_80']:>6.1%}  {cal_flag:>4s}")

    n_models = len(all_metadata["models"]) * 4  # mean + q10 + q50 + q90
    n_cals   = sum(1 for info in all_metadata["models"].values() if info["use_calibration"])
    print(f"\n  Total: {n_models} modelos + {n_cals} calibradores + 1 encoder salvos")
    print(f"  Diretório: {DATA_DIR}")

    # --- Resumo da linha dinâmica ---
    print(f"\n{'═' * 62}")
    print(f"  LINHA DINÂMICA: corners_atual + (90-min)/10  →  arred. .0/.5")
    print(f"  Odds: 1.83  |  Break-even: {1/1.83:.1%}")
    print(f"{'═' * 62}")
    print(f"  {'Min':>4s}  {'Linha méd':>10s}  {'Acurácia':>9s}  {'ROI':>8s}  {'Brier':>7s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")
    for m, info in all_metadata["models"].items():
        if "dynamic_line_accuracy" not in info:
            continue
        remaining = 90 - m
        linha_formula = f"+{remaining/10:.1f}"
        roi_str = f"{info['dynamic_line_roi']:+.1%}"
        print(f"  {m:>4d}  {linha_formula:>10s}  "
              f"{info['dynamic_line_accuracy']:>8.1%}  "
              f"{roi_str:>8s}  "
              f"{info['dynamic_line_brier']:>7.4f}")
    print(f"{'═' * 62}")

except ImportError as e:
    print(f"  Pacote não instalado ({e}). Instale: pip install xgboost scikit-learn joblib")

# %%
print("\nAnálise concluída!")
