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

            # Totais agregados
            feat["saves_total"] = ((last.get("saves_home") or 0) +
                                    (last.get("saves_away") or 0))
            feat["shots_total"] = total_shots
            feat["offsides_total"] = ((last.get("offsides_home") or 0) +
                                       (last.get("offsides_away") or 0))
            feat["cards_total"] = ((last.get("yellow_cards_home") or 0) +
                                    (last.get("yellow_cards_away") or 0) +
                                    (last.get("red_cards_home") or 0) +
                                    (last.get("red_cards_away") or 0))
            feat["goal_kicks_total"] = ((last.get("goal_kicks_home") or 0) +
                                         (last.get("goal_kicks_away") or 0))

            # Eficiência ofensiva: escanteios por tiro de meta
            feat["corners_per_goal_kick"] = round(
                c_total / max(feat["goal_kicks_total"], 1), 4)

            # Placar do intervalo (para snapshots > 45 min)
            if snap_min > 45:
                feat["ht_score_home"] = pano.get("ht_score_home")
                feat["ht_score_away"] = pano.get("ht_score_away")
            else:
                feat["ht_score_home"] = None
                feat["ht_score_away"] = None

            # ----------------------------------------------------------------
            # Game state avançado
            # ----------------------------------------------------------------
            feat["is_home_losing"] = int(score_h < score_a)
            feat["is_away_losing"] = int(score_a < score_h)
            _is_losing = int(score_h != score_a)

            # Pressão do time perdendo relativa ao total de ataques
            _total_attacks = (last.get("attacks_home") or 0) + (last.get("attacks_away") or 0)
            _total_dangerous = ((last.get("dangerous_attacks_home") or 0) +
                                (last.get("dangerous_attacks_away") or 0))
            if score_h < score_a:
                _losing_attacks   = last.get("attacks_home") or 0
                _losing_dangerous = last.get("dangerous_attacks_home") or 0
            elif score_a < score_h:
                _losing_attacks   = last.get("attacks_away") or 0
                _losing_dangerous = last.get("dangerous_attacks_away") or 0
            else:
                _losing_attacks   = 0
                _losing_dangerous = 0

            feat["losing_team_attack_share"]    = round(_losing_attacks / max(_total_attacks, 1), 4)
            feat["losing_team_dangerous_ratio"] = round(_losing_dangerous / max(_total_dangerous, 1), 4)

            # Urgency: quanto mais tempo perdendo faltando + menos tempo → mais urgente
            feat["urgency_index"]    = _is_losing * (90 - snap_min)
            feat["urgency_weighted"] = round(_is_losing / max(90 - snap_min, 1), 4)

            # ----------------------------------------------------------------
            # Pressão ofensiva últimos 5 / 10 min (ataques e chutes)
            # ----------------------------------------------------------------
            _att_last5_h  = _last_n_minutes(until, snap_min, 5,  "attacks_home")
            _att_last5_a  = _last_n_minutes(until, snap_min, 5,  "attacks_away")
            _att_last10_h = _last_n_minutes(until, snap_min, 10, "attacks_home")
            _att_last10_a = _last_n_minutes(until, snap_min, 10, "attacks_away")
            _da_last5_h   = _last_n_minutes(until, snap_min, 5,  "dangerous_attacks_home")
            _da_last5_a   = _last_n_minutes(until, snap_min, 5,  "dangerous_attacks_away")
            _da_last10_h  = _last_n_minutes(until, snap_min, 10, "dangerous_attacks_home")
            _da_last10_a  = _last_n_minutes(until, snap_min, 10, "dangerous_attacks_away")
            _son_last5_h  = _last_n_minutes(until, snap_min, 5,  "shots_on_target_home")
            _son_last5_a  = _last_n_minutes(until, snap_min, 5,  "shots_on_target_away")
            _soff_last5_h = _last_n_minutes(until, snap_min, 5,  "shots_off_target_home")
            _soff_last5_a = _last_n_minutes(until, snap_min, 5,  "shots_off_target_away")

            feat["attacks_last_5min"]          = (_att_last5_h  or 0) + (_att_last5_a  or 0)
            feat["attacks_last_10min"]         = (_att_last10_h or 0) + (_att_last10_a or 0)
            feat["dangerous_attacks_last_5min"]  = (_da_last5_h  or 0) + (_da_last5_a  or 0)
            feat["dangerous_attacks_last_10min"] = (_da_last10_h or 0) + (_da_last10_a or 0)
            feat["shots_last_5min"] = ((_son_last5_h or 0) + (_son_last5_a or 0) +
                                       (_soff_last5_h or 0) + (_soff_last5_a or 0))

            # Pressure acceleration: ritmo atual (last 5 min) vs ritmo médio do jogo
            _avg_att_rate = _total_attacks / max(snap_min, 1)
            feat["pressure_acceleration"] = round(
                (feat["attacks_last_5min"] / 5) / max(_avg_att_rate, 0.01), 4)

            # Dominância por time (share de ataques e ataques perigosos)
            feat["attack_share_home"]    = round((last.get("attacks_home") or 0) / max(_total_attacks, 1), 4)
            feat["attack_share_away"]    = round((last.get("attacks_away") or 0) / max(_total_attacks, 1), 4)
            feat["dangerous_share_home"] = round((last.get("dangerous_attacks_home") or 0) / max(_total_dangerous, 1), 4)
            feat["dangerous_share_away"] = round((last.get("dangerous_attacks_away") or 0) / max(_total_dangerous, 1), 4)

            # Momentum ratios: atividade recente / total acumulado
            feat["corners_momentum_ratio"] = round(feat["corners_last_10min"] / max(c_total, 1), 4)
            feat["attacks_momentum_ratio"] = round(feat["attacks_last_10min"] / max(_total_attacks, 1), 4)

            # Qualidade de chute: chutes por ataque perigoso
            feat["shots_per_dangerous_attack"] = round(
                total_shots / max(_total_dangerous, 1), 4)

            # ----------------------------------------------------------------
            # Time since last corner / shot / dangerous attack (streaks)
            # ----------------------------------------------------------------
            def _time_since_last_event(df_until, col_home, col_away, current_min):
                """Minutos desde o último evento que incrementou a contagem acumulada."""
                if col_home not in df_until.columns:
                    return None
                total_s = df_until[col_home].fillna(0)
                if col_away in df_until.columns:
                    total_s = total_s + df_until[col_away].fillna(0)
                diff = total_s.diff()
                events = df_until[diff > 0]
                if events.empty:
                    return current_min  # nenhum evento ainda
                last_ev_min = events.iloc[-1].get("minute")
                if last_ev_min is None:
                    return None
                return int(current_min - last_ev_min)

            feat["time_since_last_corner"] = _time_since_last_event(
                until, "corners_home", "corners_away", snap_min)
            feat["time_since_last_shot"] = _time_since_last_event(
                until, "shots_on_target_home", "shots_on_target_away", snap_min)
            feat["time_since_last_dangerous_attack"] = _time_since_last_event(
                until, "dangerous_attacks_home", "dangerous_attacks_away", snap_min)

            # ----------------------------------------------------------------
            # Não-linearidade do tempo
            # ----------------------------------------------------------------
            feat["snap_minute_sq"] = snap_min ** 2
            feat["snap_minute_sqrt"]  = round(snap_min ** 0.5, 4)
            feat["snap_minute_log"]   = round(np.log(max(snap_min, 1)), 4)
            feat["remaining_time_sq"] = (90 - snap_min) ** 2
            feat["time_ratio"]        = round(snap_min / 90, 4)
            feat["phase_of_game"]  = (0 if snap_min <= 30 else
                                      1 if snap_min <= 60 else
                                      2 if snap_min <= 75 else 3)
            feat["is_last_15min"]      = int(snap_min >= 75)
            feat["losing_in_last_15"]  = int(snap_min >= 75 and score_h != score_a)

            # ----------------------------------------------------------------
            # Taxas de escanteio por estado do placar
            # ----------------------------------------------------------------
            if score_h < score_a:           # home losing
                _losing_corners  = c_home
                _winning_corners = c_away
            elif score_a < score_h:         # away losing
                _losing_corners  = c_away
                _winning_corners = c_home
            else:                           # draw
                _losing_corners  = 0
                _winning_corners = 0

            feat["losing_team_corners_rate"]  = round(_losing_corners  / max(snap_min, 1), 4)
            feat["winning_team_corners_rate"] = round(_winning_corners / max(snap_min, 1), 4)

            # Pressão perigosa do time perdendo por minuto
            if score_h < score_a:
                _losing_da = last.get("dangerous_attacks_home") or 0
            elif score_a < score_h:
                _losing_da = last.get("dangerous_attacks_away") or 0
            else:
                _losing_da = 0
            feat["pressure_when_losing"] = round(_losing_da / max(snap_min, 1), 4)

            # ----------------------------------------------------------------
            # Qualidade da pressão e distribuição de campo
            # ----------------------------------------------------------------
            feat["pressure_ratio"]    = round(_total_dangerous / max(_total_attacks, 1), 4)
            _da_h = last.get("dangerous_attacks_home") or 0
            _da_a = last.get("dangerous_attacks_away") or 0
            feat["field_tilt_proxy"]  = round(_da_h / max(_da_h + _da_a, 1), 4)

            # ----------------------------------------------------------------
            # Aceleração de escanteios: last_5 - prev_5 (janela anterior)
            # ----------------------------------------------------------------
            feat["acceleration_corners"] = round(
                2 * feat["corners_last_5min"] - feat["corners_last_10min"], 4)

            # ----------------------------------------------------------------
            # Momentum score: soma ponderada de eventos nos últimos 15 min
            # (corners peso 3, dangerous peso 2, attacks peso 1)
            # ----------------------------------------------------------------
            _c_last15  = (feat["corners_last_15_home"] or 0) + \
                         (_last_n_minutes(until, snap_min, 15, "corners_away") or 0)
            _da_last15 = (_last_n_minutes(until, snap_min, 15, "dangerous_attacks_home") or 0) + \
                         (_last_n_minutes(until, snap_min, 15, "dangerous_attacks_away") or 0)
            _att_last15 = (_last_n_minutes(until, snap_min, 15, "attacks_home") or 0) + \
                          (_last_n_minutes(until, snap_min, 15, "attacks_away") or 0)
            feat["momentum_score"] = round(3 * _c_last15 + 2 * _da_last15 + _att_last15, 4)

            # ----------------------------------------------------------------
            # Game regime: LOW / NORMAL / HIGH (0/1/2)
            # Baseado em ritmo de escanteios + ataques perigosos + chutes
            # ----------------------------------------------------------------
            _c_rate   = feat["corners_rate_per_min"]
            _da_rate  = feat["dangerous_attacks_rate"]
            _sh_rate  = total_shots / max(snap_min, 1)
            _regime_score = _c_rate * 10 + _da_rate + _sh_rate * 0.5
            feat["game_regime"] = (0 if _regime_score < 0.8 else
                                   1 if _regime_score < 1.8 else 2)

            # ----------------------------------------------------------------
            # Dominância ofensiva e assimetria
            # ----------------------------------------------------------------
            feat["dangerous_dominance"] = _da_h - _da_a   # + = home domina
            feat["corner_dominance"]    = c_home - c_away  # equivale a corners_diff
            feat["one_sided_game"]      = abs(c_home - c_away)

            # ----------------------------------------------------------------
            # Expected corners pelo minuto + resíduo + flags
            # ----------------------------------------------------------------
            _league_avg_v = league_avg_map.get(str(event_id)) if league_avg_map else None
            _exp_by_min = (round(_league_avg_v * snap_min / 90, 4)
                           if _league_avg_v is not None else None)
            feat["expected_corners_by_minute"] = _exp_by_min
            if _exp_by_min is not None:
                _res = round(c_total - _exp_by_min, 4)
                feat["residual_corners"]        = _res
                feat["overperformance_flag"]    = int(_res >  1.5)
                feat["underperformance_flag"]   = int(_res < -1.5)
            else:
                feat["residual_corners"]        = None
                feat["overperformance_flag"]    = 0
                feat["underperformance_flag"]   = 0

            # ----------------------------------------------------------------
            # Intensidade 1º vs 2º tempo (só snap > 45)
            # ----------------------------------------------------------------
            if snap_min > 45 and "first_half_corners" in feat:
                _fh = feat.get("first_half_corners") or 0
                _fh_rate  = _fh / 45
                _sh_rate2 = feat.get("second_half_rate") or 0
                feat["intensity_drop"]           = round(_fh_rate - _sh_rate2, 4)
                feat["second_half_decay_factor"] = round(
                    _sh_rate2 / max(_fh_rate, 0.01), 4)
                _ff = feat.get("fatigue_factor") or 1.0
                feat["tempo_adjusted_rate"] = round(feat["corners_rate_per_min"] * _ff, 4)
            else:
                feat["intensity_drop"]           = None
                feat["second_half_decay_factor"] = None
                feat["tempo_adjusted_rate"]      = None

            # ----------------------------------------------------------------
            # Novas features de pressão e ritmo
            # ----------------------------------------------------------------

            # pressure_diff: diferença bruta de ataques perigosos (home - away)
            # (nota: dangerous_dominance já existe com o mesmo valor)
            feat["pressure_diff"] = _da_h - _da_a

            # pressure_dominance_ratio: assimetria lateral de pressão perigosa
            feat["pressure_dominance_ratio"] = round(_da_h / (_da_a + 1), 4)

            # rolling_pressure_5 / 10: taxa de ataques perigosos nos últimos 5/10 min
            feat["rolling_pressure_5"]  = round(feat["dangerous_attacks_last_5min"]  / 5,  4)
            feat["rolling_pressure_10"] = round(feat["dangerous_attacks_last_10min"] / 10, 4)

            # da_pressure_acceleration: aceleração baseada em ataques perigosos
            feat["da_pressure_acceleration"] = round(
                feat["rolling_pressure_5"] - feat["rolling_pressure_10"], 4)

            # losing_team_pressure_ratio: share perigoso do time perdendo
            feat["losing_team_pressure_ratio"] = round(
                _losing_da / (_total_dangerous + 1), 4)

            # urgency: intensidade de reação × momento da partida
            feat["urgency"] = round(
                abs(score_h - score_a) * (snap_min / 90), 4)

            # final_pressure: pressão recente relativa ao tempo restante
            feat["final_pressure"] = round(
                feat["dangerous_attacks_last_10min"] / (90 - snap_min + 1), 4)

            # activity_spike: pico de atividade nos últimos 5 min
            feat["activity_spike"] = (
                feat["corners_last_5min"] +
                feat["shots_last_5min"] +
                feat["dangerous_attacks_last_5min"]
            )

            # corner_drought_pressure: pressão acumulada durante jejum de escanteios
            _tslc = feat.get("time_since_last_corner") or 0
            feat["corner_drought_pressure"] = round(_tslc * feat["pressure_ratio"], 4)

            # corner_conversion: eficiência de converter pressão perigosa em escanteios
            feat["corner_conversion"] = round(c_total / (_total_dangerous + 1), 4)

            # wasted_pressure: ataques perigosos não convertidos em chutes
            feat["wasted_pressure"] = max(_total_dangerous - total_shots, 0)

            # is_high_pace / is_low_pace: ritmo atual vs média da liga
            _league_rate = (_league_avg_v / 90) if _league_avg_v else None
            if _league_rate:
                feat["is_high_pace"] = int(feat["corners_rate_per_min"] > _league_rate)
                feat["is_low_pace"]  = int(feat["corners_rate_per_min"] < _league_rate * 0.7)
            else:
                feat["is_high_pace"] = 0
                feat["is_low_pace"]  = 0

            # pace_shift: ritmo recente (last 10) vs ritmo do 1º tempo
            if snap_min > 45 and feat.get("first_half_corners") is not None:
                _fh_rate_for_shift = (feat.get("first_half_corners") or 0) / 45
                feat["pace_shift"] = round(feat["corners_rate_last_10"] - _fh_rate_for_shift, 4)
            else:
                feat["pace_shift"] = 0

            # --- Features pré-jogo do panorama ---
            # Odds pré-jogo
            for col in ["corners_line", "corners_over_odds", "corners_under_odds",
                        "asian_corners_line", "asian_corners_home_odds", "asian_corners_away_odds",
                        "odds_home_win", "odds_draw", "odds_away_win",
                        "goals_line", "goals_over_odds", "goals_under_odds",
                        "btts_yes_odds", "btts_no_odds"]:
                feat[col] = pano.get(col)

            # Odds ao vivo (se disponíveis)
            for col in ["live_corners_line", "live_corners_over_odds",
                        "live_corners_under_odds"]:
                feat[col] = pano.get(col)

            # Divergência mercado vs expectativa histórica
            _cl = feat.get("corners_line")
            _me = feat.get("match_expected_corners")
            if _cl is not None and _me is not None:
                feat["line_diff_vs_expected"] = round(float(_cl) - float(_me), 4)
            else:
                feat["line_diff_vs_expected"] = None

            # Stats adicionais do event/view
            for col in ["throw_ins_home_total", "throw_ins_away_total",
                        "tackles_home_total", "tackles_away_total"]:
                feat[col] = pano.get(col)

            # Days rest (do histórico dos times)
            feat["days_rest_home"] = hist.get("days_rest_home")
            feat["days_rest_away"] = hist.get("days_rest_away")

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
# =============================================================================
# 3 / 4. CACHE DE FEATURES
#
# Se features_ml.parquet já existe E os dados de origem não mudaram desde a
# última geração, carrega direto do disco — evita reconstruir do zero.
#
# Para forçar rebuild: delete o arquivo ou passe --rebuild na linha de comando.
#   python betsapi_corners_analysis.py --rebuild
# =============================================================================
import sys as _sys
_FEATURES_PATH = DATA_DIR / "features_ml.parquet"
_FORCE_REBUILD = "--rebuild" in _sys.argv
_FORCE_RETUNE  = "--retune"  in _sys.argv
_HPARAMS_PATH  = DATA_DIR / "optuna_hparams.joblib"

def _snap_mtime():
    """Retorna o maior mtime entre os dois parquets de origem."""
    t1 = (DATA_DIR / "snapshots_por_minuto.parquet").stat().st_mtime
    t2 = (DATA_DIR / "panorama_jogos.parquet").stat().st_mtime
    return max(t1, t2)

def _features_fresh():
    """True se features_ml.parquet existe e é mais recente que os dados brutos."""
    if not _FEATURES_PATH.exists():
        return False
    return _FEATURES_PATH.stat().st_mtime >= _snap_mtime()

if not _FORCE_REBUILD and _features_fresh():
    print(f"\nCarregando features do cache: {_FEATURES_PATH}")
    df_features = pd.read_parquet(_FEATURES_PATH)
    print(f"  {len(df_features):,} linhas | {df_features['event_id'].nunique():,} jogos  (use --rebuild para regenerar)")
else:
    if _FORCE_REBUILD:
        print("\n--rebuild solicitado: reconstruindo features do zero...")
    else:
        print("\nfeatures_ml.parquet não existe ou está desatualizado — construindo...")
    df_features = build_live_features(df_snap, df_pano, df_team_hist, SNAPSHOT_MINUTES,
                                       league_avg_map=league_avg,
                                       league_std_map=league_std)
    print(f"  {len(df_features):,} linhas | {df_features['event_id'].nunique():,} jogos")

    # Limpeza básica
    df_features = df_features[df_features["target_corners_total"].notna()]
    df_features = df_features[df_features["target_corners_total"] >= 0]
    df_features["target_corners_remaining"] = df_features["target_corners_remaining"].clip(lower=0)

    # Salva no disco
    df_features.to_parquet(_FEATURES_PATH, index=False)
    df_features.to_csv(DATA_DIR / "features_ml.csv", index=False)
    print(f"  Cache salvo → {_FEATURES_PATH}  ({_FEATURES_PATH.stat().st_size / 1e6:.1f} MB)")

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

# Target encoding será aplicado DENTRO do loop por minuto, após o split temporal,
# para evitar data leakage do conjunto de teste.
print(f"\nTarget encoding ({ENCODE_COLS}) será aplicado por minuto após o split temporal.")

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

# (features já salvas no bloco de cache acima)

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
    from scipy.stats import poisson as sp_poisson
    from scipy.stats import nbinom as sp_nbinom
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import joblib

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        _OPTUNA = True
    except ImportError:
        _OPTUNA = False
        print("  (optuna não instalado — usando hiperparâmetros fixos. pip install optuna)")

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
        "shots_total",
        # Defesas, impedimentos, tiros de meta
        "saves_home", "saves_away", "saves_total",
        "offsides_home", "offsides_away", "offsides_total",
        "goal_kicks_home", "goal_kicks_away", "goal_kicks_total",
        "corners_per_goal_kick",
        # Cartões e faltas (individuais + agregados)
        "yellow_cards_home", "yellow_cards_away",
        "fouls_home", "fouls_away",
        "fouls_diff", "fouls_total",
        "cards_total",
        # Placar e contexto
        "score_home", "score_away",
        "score_diff", "total_red_cards", "red_card_diff",
        "is_draw",
        # Placar do intervalo (disponível para snap > 45)
        "ht_score_home", "ht_score_away",
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
        # Conversão / eficiência
        "corners_per_dangerous_attack", "shots_per_corner",
        # Odds pré-jogo
        "corners_line", "corners_over_odds", "corners_under_odds",
        "asian_corners_line", "asian_corners_home_odds", "asian_corners_away_odds",
        "odds_home_win", "odds_draw", "odds_away_win",
        "goals_line", "goals_over_odds", "goals_under_odds",
        "btts_yes_odds", "btts_no_odds",
        # Odds ao vivo
        "live_corners_line", "live_corners_over_odds", "live_corners_under_odds",
        # Divergência mercado vs expectativa
        "line_diff_vs_expected",
        # Stats do event/view
        "throw_ins_home_total", "throw_ins_away_total",
        "tackles_home_total", "tackles_away_total",
        # Descanso
        "days_rest_home", "days_rest_away",
        # Qualidade dos dados
        "n_snap_minutes",
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
        # Game state avançado
        "is_home_losing", "is_away_losing",
        "losing_team_attack_share", "losing_team_dangerous_ratio",
        "urgency_index", "urgency_weighted",
        # Pressão ofensiva últimos 5/10 min
        "attacks_last_5min", "attacks_last_10min",
        "dangerous_attacks_last_5min", "dangerous_attacks_last_10min",
        "shots_last_5min",
        "pressure_acceleration",
        # Dominância por time
        "attack_share_home", "attack_share_away",
        "dangerous_share_home", "dangerous_share_away",
        # Momentum ratios
        "corners_momentum_ratio", "attacks_momentum_ratio",
        # Qualidade de chute
        "shots_per_dangerous_attack",
        # Streaks (tempo desde último evento)
        "time_since_last_corner", "time_since_last_shot",
        "time_since_last_dangerous_attack",
        # Não-linearidade do tempo
        "snap_minute_sq", "snap_minute_sqrt", "snap_minute_log",
        "remaining_time_sq", "time_ratio",
        "phase_of_game",
        "is_last_15min", "losing_in_last_15",
        # Taxas por estado de placar
        "losing_team_corners_rate", "winning_team_corners_rate",
        "pressure_when_losing",
        # Qualidade da pressão e distribuição de campo
        "pressure_ratio", "field_tilt_proxy",
        # Aceleração e momentum
        "acceleration_corners", "momentum_score",
        # Game regime
        "game_regime",
        # Dominância ofensiva
        "dangerous_dominance", "corner_dominance", "one_sided_game",
        # Esperado vs real (resíduo)
        "expected_corners_by_minute", "residual_corners",
        "overperformance_flag", "underperformance_flag",
        # Dinâmica 1º/2º tempo
        "intensity_drop", "second_half_decay_factor", "tempo_adjusted_rate",
        # Pressão e ritmo (novas)
        "pressure_diff", "pressure_dominance_ratio",
        "rolling_pressure_5", "rolling_pressure_10", "da_pressure_acceleration",
        "losing_team_pressure_ratio",
        "urgency",
        "final_pressure",
        "activity_spike",
        "corner_drought_pressure",
        "corner_conversion",
        "wasted_pressure",
        "is_high_pace", "is_low_pace",
        "pace_shift",
    ]

    # Features de momentum (só disponíveis para minutos > 15)
    MOMENTUM_FEATURE_COLS = [
        "delta_corners_total_so_far", "delta_corners_home_so_far",
        "delta_corners_away_so_far", "delta_corners_rate_per_min",
        "delta_corners_last_5min", "delta_corners_last_10min",
        "delta_corners_rate_last_5", "delta_corners_rate_last_10",
        "delta_dangerous_attacks_home", "delta_dangerous_attacks_away",
        "delta_attacks_home", "delta_attacks_away",
        "delta_shots_on_target_home", "delta_shots_on_target_away",
        "delta_possession_home_avg",
    ]

    TARGET = "target_corners_total"

    # ------------------------------------------------------------------
    # Verificação de integridade do cache
    #
    # 1. Exclui features criadas fora do build_live_features:
    #    - *_target_enc  → criadas no loop de treino após split temporal
    #    - delta_*       → criadas no bloco de momentum após build
    # 2. Features deriváveis de colunas existentes → patch incremental
    # 3. Só faz full rebuild se faltar feature que precisa de dados brutos
    # ------------------------------------------------------------------
    _SKIP_CACHE_CHECK = {c for c in (BASE_FEATURE_COLS + MOMENTUM_FEATURE_COLS)
                         if c.endswith("_target_enc") or c.startswith("delta_")}

    # Features que podem ser calculadas incrementalmente a partir do cache
    _PATCHABLE: dict[str, callable] = {
        "snap_minute_sqrt":  lambda df: df["snap_minute"].apply(lambda x: round(x ** 0.5, 4)),
        "snap_minute_log":   lambda df: df["snap_minute"].apply(lambda x: round(np.log(max(x, 1)), 4)),
        "remaining_time_sq": lambda df: (90 - df["snap_minute"]) ** 2,
        "time_ratio":        lambda df: (df["snap_minute"] / 90).round(4),
    }

    _all_expected = set(BASE_FEATURE_COLS + MOMENTUM_FEATURE_COLS) - _SKIP_CACHE_CHECK
    _missing_feats = _all_expected - set(df_features.columns)

    if _missing_feats:
        # Tenta patch incremental para features simples
        _patched = set()
        for _f in list(_missing_feats):
            if _f in _PATCHABLE:
                _src_col = "snap_minute"
                if _src_col in df_features.columns:
                    df_features[_f] = _PATCHABLE[_f](df_features)
                    _patched.add(_f)

        _still_missing = _missing_feats - _patched
        if _patched:
            print(f"\n  Cache: +{len(_patched)} feature(s) adicionadas incrementalmente: "
                  f"{', '.join(sorted(_patched))}")
            df_features.to_parquet(_FEATURES_PATH, index=False)
            print(f"  Cache atualizado → {_FEATURES_PATH}")

        if _still_missing:
            print(f"\n  Cache desatualizado — {len(_still_missing)} feature(s) precisam de rebuild:")
            for _f in sorted(_still_missing):
                print(f"    • {_f}")
            print("  Reconstruindo features do zero...\n")
            df_features = build_live_features(
                df_snap, df_pano, df_team_hist, SNAPSHOT_MINUTES,
                league_avg_map=league_avg, league_std_map=league_std,
            )
            df_features = df_features[df_features["target_corners_total"].notna()]
            df_features = df_features[df_features["target_corners_total"] >= 0]
            df_features["target_corners_remaining"] = df_features["target_corners_remaining"].clip(lower=0)
            df_features.to_parquet(_FEATURES_PATH, index=False)
            df_features.to_csv(DATA_DIR / "features_ml.csv", index=False)
            print(f"  Cache atualizado → {_FEATURES_PATH}  "
                  f"({_FEATURES_PATH.stat().st_size / 1e6:.1f} MB)")
            print(f"  {len(df_features):,} linhas | {df_features['event_id'].nunique():,} jogos")

    def prepare_features(
        df: pd.DataFrame,
        feature_cols: list[str],
        medians: dict | None = None,
        available_override: list[str] | None = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Filtra features existentes, remove ≥99% NaN, preenche NaN restantes.

        medians: dict {col: valor} calculado APENAS no treino para evitar leakage.
                 Se None, calcula na própria df (usado apenas para inferir available).
        available_override: lista de colunas já definida (usada em cal/test para
                            respeitar as mesmas colunas do treino).
        """
        if available_override is not None:
            available = [c for c in available_override if c in df.columns]
        else:
            available = [c for c in feature_cols if c in df.columns]
            # Remove colunas quase vazias (avaliado no treino; aqui serve de filtro inicial)
            null_pcts = df[available].isnull().mean()
            available = [c for c in available if null_pcts[c] < 0.99]

        df_out = df[available + [TARGET]].copy()
        df_out = df_out.dropna(subset=[TARGET])

        # Preenche NaN: históricas/liga/encoding → mediana do treino; ao vivo → 0
        fill_med = [c for c in available if c.startswith(("hist_", "league_"))
                    or c.endswith("_target_enc")]
        fill_zero = [c for c in available if c not in fill_med]

        for c in fill_med:
            fill_val = medians[c] if (medians and c in medians) else df_out[c].median()
            df_out[c] = df_out[c].fillna(fill_val)
        for c in fill_zero:
            df_out[c] = df_out[c].fillna(0)

        return available, df_out

    # --- Armazenamento de resultados e artefatos ---
    all_metadata: dict = {"snapshot_minutes": SNAPSHOT_MINUTES, "models": {}}

    print(f"\n  Dataset total: {len(df_features):,} amostras")
    print(f"  Split temporal: 60% treino / 20% calibração / 20% teste (ordem cronológica)")

    for snap_min in SNAPSHOT_MINUTES:
        print(f"\n{'─' * 62}")
        print(f"  ⚽ MINUTO {snap_min}")
        print(f"{'─' * 62}")

        df_min = df_features[df_features["snap_minute"] == snap_min].copy()

        # --- Split temporal: ordena por data de jogo antes de dividir ---
        if "kickoff_dt" in df_min.columns:
            df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
        n_total_min = len(df_min)
        n_test_sz   = int(n_total_min * 0.20)
        n_cal_sz    = int((n_total_min - n_test_sz) * 0.25)  # 25% do trainval ≈ 20% do total
        n_train_sz  = n_total_min - n_test_sz - n_cal_sz

        df_train_raw = df_min.iloc[:n_train_sz].copy()
        df_cal_raw   = df_min.iloc[n_train_sz:n_train_sz + n_cal_sz].copy()
        df_test_raw  = df_min.iloc[n_train_sz + n_cal_sz:].copy()

        # --- Target encoding: fit APENAS no treino, transforma todos os splits ---
        encode_cols_avail = [c for c in ENCODE_COLS if c in df_min.columns]
        if encode_cols_avail:
            te_min = TargetEncoderSmoothed(
                cols=encode_cols_avail, target_col=TARGET, smoothing=10
            )
            te_min.fit(df_train_raw)
            df_train_raw = te_min.transform(df_train_raw)
            df_cal_raw   = te_min.transform(df_cal_raw)
            df_test_raw  = te_min.transform(df_test_raw)
        else:
            te_min = None

        # Minuto 15 não tem momentum; demais sim
        feat_cols = BASE_FEATURE_COLS if snap_min == 15 else BASE_FEATURE_COLS + MOMENTUM_FEATURE_COLS

        # --- Determina features disponíveis e medianas usando APENAS o treino ---
        available_train, df_train_clean = prepare_features(df_train_raw, feat_cols)

        if len(df_train_clean) < 80:
            print(f"  ⚠ Dados insuficientes no treino ({len(df_train_clean)} amostras). Pulando.")
            continue

        fill_med_cols = [c for c in available_train
                         if c.startswith(("hist_", "league_")) or c.endswith("_target_enc")]
        train_medians = {c: df_train_clean[c].median() for c in fill_med_cols}

        # Aplica as mesmas colunas e medianas do treino ao cal e test
        _, df_cal_clean  = prepare_features(df_cal_raw,  feat_cols,
                                             medians=train_medians,
                                             available_override=available_train)
        _, df_test_clean = prepare_features(df_test_raw, feat_cols,
                                             medians=train_medians,
                                             available_override=available_train)

        available = available_train

        if len(df_test_clean) < 20:
            print(f"  ⚠ Dados insuficientes no teste ({len(df_test_clean)} amostras). Pulando.")
            continue

        X_train = df_train_clean[available]
        y_train = df_train_clean[TARGET]
        X_cal   = df_cal_clean[available]
        y_cal   = df_cal_clean[TARGET]
        X_test  = df_test_clean[available]
        y_test  = df_test_clean[TARGET]

        print(f"  Amostras: treino={len(X_train):,}  cal={len(X_cal):,}  teste={len(X_test):,}")
        print(f"  Features: {len(available)}")

        # ==================================================================
        # 6a. Modelo principal — Optuna tuning com cache de hiperparâmetros
        #
        # Cache: dados_escanteios/optuna_hparams.joblib
        #   dict {snap_min: {"params": {...}, "mae_cal": float}}
        # Fluxo:
        #   1ª execução ou --retune  → roda Optuna (50 trials) e salva
        #   Demais execuções         → carrega params do cache (~0s)
        # ==================================================================
        # Carrega cache existente (ou vazio)
        if _HPARAMS_PATH.exists():
            _hparams_cache = joblib.load(_HPARAMS_PATH)
        else:
            _hparams_cache = {}

        _cached = _hparams_cache.get(snap_min)
        _run_optuna = (_OPTUNA and len(X_train) >= 200
                       and (_FORCE_RETUNE or _cached is None))

        if _run_optuna:
            def _xgb_trial(trial):
                _p = dict(
                    n_estimators     = trial.suggest_int("n_estimators", 200, 800),
                    max_depth        = trial.suggest_int("max_depth", 3, 8),
                    learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                    subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight = trial.suggest_int("min_child_weight", 3, 15),
                    reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 1.0),
                    reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
                    random_state=42, verbosity=0, early_stopping_rounds=30,
                )
                _m = xgb.XGBRegressor(**_p)
                _m.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
                return mean_absolute_error(y_cal, _m.predict(X_cal))

            _study = _optuna.create_study(
                direction="minimize",
                sampler=_optuna.samplers.TPESampler(seed=42),
            )
            _study.optimize(_xgb_trial, n_trials=50, show_progress_bar=False)
            _best_xgb = dict(_study.best_params)
            _best_xgb.update({"random_state": 42, "verbosity": 0, "early_stopping_rounds": 30})
            # Salva no cache
            _hparams_cache[snap_min] = {
                "params": _best_xgb,
                "mae_cal": round(_study.best_value, 4),
            }
            joblib.dump(_hparams_cache, _HPARAMS_PATH)
            print(f"  Optuna (50 trials): MAE(cal)={_study.best_value:.3f}  "
                  f"depth={_best_xgb['max_depth']}  lr={_best_xgb['learning_rate']:.4f}  [salvo]")

        elif _cached is not None:
            _best_xgb = _cached["params"]
            print(f"  Hparams (cache): MAE(cal)={_cached['mae_cal']:.3f}  "
                  f"depth={_best_xgb['max_depth']}  lr={_best_xgb['learning_rate']:.4f}")

        else:
            _best_xgb = dict(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0, early_stopping_rounds=30,
            )
            print(f"  Hparams (default): optuna não disponível")

        model_mean = xgb.XGBRegressor(**_best_xgb)
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
        # 6c. Quantile regression (P10, P50, P90)
        #
        # Treinado ANTES de 6f para fornecer sigma heteroscedástico por jogo.
        # NÃO aplica calibração isotônica — isso colapsaria os intervalos.
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
                reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30,
            )
            model_q.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
            quantile_models[q_name] = model_q

        p10_test = quantile_models["q10"].predict(X_test)
        p50_test = quantile_models["q50"].predict(X_test)
        p90_test = quantile_models["q90"].predict(X_test)
        p10_cal  = quantile_models["q10"].predict(X_cal)
        p90_cal  = quantile_models["q90"].predict(X_cal)

        coverage       = ((y_test.values >= p10_test) & (y_test.values <= p90_test)).mean()
        interval_width = (p90_test - p10_test).mean()
        mae_p50        = mean_absolute_error(y_test, p50_test)
        below_p10      = (y_test.values < p10_test).mean()
        above_p90      = (y_test.values > p90_test).mean()

        print(f"\n  Quantile regression:")
        print(f"    P50 MAE           : {mae_p50:.3f}")
        print(f"    Intervalo P10-P90 : largura média = {interval_width:.1f} corners")
        print(f"    Cobertura P10-P90 : {coverage:.1%} (ideal: ~80%)")
        print(f"    Abaixo do P10     : {below_p10:.1%} (ideal: ~10%)")
        print(f"    Acima do P90      : {above_p90:.1%} (ideal: ~10%)")

        # ==================================================================
        # 6f. Avaliação da linha dinâmica — sistema completo de betting
        #
        # Melhorias vs versão anterior:
        #   (a) Linha por liga: corners_atual + remaining × (league_avg/90)
        #   (b) Sigma heteroscedástico: (p90-p10)/(2×1.28) por jogo
        #   (c) Logística multi-feature (pred, diff, diff_norm, minute, contexto)
        #   (d) XGBoost classificador direto P(over/under linha)
        #   (e) Threshold selecionado no CAL set — sem leakage do test
        #   (f) Edge mínimo: P(over) > break-even + MIN_EDGE
        #   (g) Under simétrico
        #   (h) ROI com odds reais por jogo quando disponíveis
        #   (i) Breakdown por contexto (game_regime, phase, faixas de linha)
        # ==================================================================
        ODDS_OVER  = 1.83
        ODDS_UNDER = 1.83   # default; substituído por odds reais quando disponível
        BREAKEVEN  = 1.0 / ODDS_OVER
        MIN_EDGE   = 0.02   # só aposta se P(over) > break-even + 2%
        remaining_minutes = 90 - snap_min

        # ---- (a) Linha dinâmica por liga ----
        def _dline_vec(csf_arr, la_arr, rem):
            """Usa taxa média da liga (corners/min) se disponível; fallback 0.1/min."""
            lines = []
            for csf, la in zip(csf_arr, la_arr):
                try:
                    rate = (float(la) / 90.0) if (la is not None and not np.isnan(float(la))
                                                   and float(la) > 0) else 0.1
                except (TypeError, ValueError):
                    rate = 0.1
                lines.append(np.round((csf + rem * rate) * 2) / 2)
            return np.array(lines, dtype=float)

        def _gcol(X_df, col):
            return X_df[col].values if col in X_df.columns else np.zeros(len(X_df))
        def _gcol_nan(X_df, col):
            return X_df[col].values if col in X_df.columns else np.full(len(X_df), np.nan)

        csf_test  = _gcol(X_test,  "corners_total_so_far")
        csf_cal   = _gcol(X_cal,   "corners_total_so_far")
        csf_train = _gcol(X_train, "corners_total_so_far")

        def _la_list(X_df):
            raw = _gcol_nan(X_df, "league_avg_corners")
            return [v if not np.isnan(v) else None for v in raw]

        dynamic_line       = _dline_vec(csf_test,  _la_list(X_test),  remaining_minutes)
        dynamic_line_cal   = _dline_vec(csf_cal,   _la_list(X_cal),   remaining_minutes)
        dynamic_line_train = _dline_vec(csf_train, _la_list(X_train), remaining_minutes)

        over_actual       = (y_test.values  > dynamic_line).astype(int)
        over_actual_cal   = (y_cal.values   > dynamic_line_cal).astype(int)
        over_actual_train = (y_train.values > dynamic_line_train).astype(int)

        preds_cal_best  = calibrator.predict(preds_cal_raw) if use_calibration else preds_cal_raw
        preds_cal_c     = np.clip(preds_cal_best, 0.1, 60.0)
        preds_test_c    = np.clip(preds_best,     0.1, 60.0)
        _preds_train_raw = model_mean.predict(X_train)
        preds_train_c   = np.clip(
            calibrator.predict(_preds_train_raw) if use_calibration else _preds_train_raw,
            0.1, 60.0)

        fl_test = np.floor(dynamic_line).astype(int)
        fl_cal  = np.floor(dynamic_line_cal).astype(int)

        # ---- (b) Sigma heteroscedástico (q10-q90) ----
        sigma_het_test = np.maximum(p90_test - p10_test, 1.5) / (2 * 1.28)
        sigma_het_cal  = np.maximum(p90_cal  - p10_cal,  1.5) / (2 * 1.28)

        # ---- (1) Normal heteroscedástico ----
        p_normal_test = 1.0 - sp_norm.cdf(dynamic_line,     loc=preds_test_c, scale=sigma_het_test)
        p_normal_cal  = 1.0 - sp_norm.cdf(dynamic_line_cal, loc=preds_cal_c,  scale=sigma_het_cal)

        # ---- (2) Poisson ----
        p_poisson_test = np.array([1.0 - sp_poisson.cdf(fl, mu=max(m, 0.01))
                                   for fl, m in zip(fl_test, preds_test_c)])
        p_poisson_cal  = np.array([1.0 - sp_poisson.cdf(fl, mu=max(m, 0.01))
                                   for fl, m in zip(fl_cal, preds_cal_c)])

        # ---- (3) Negative Binomial ----
        _resid_var = float(np.mean((y_cal.values - preds_cal_c) ** 2))
        _mu_mean   = float(np.mean(preds_cal_c))
        nb_r = float(np.clip(_mu_mean ** 2 / max(_resid_var - _mu_mean, 0.5), 0.5, 100.0))
        p_nb_test = np.array([1.0 - sp_nbinom.cdf(fl, n=nb_r, p=nb_r / (nb_r + max(m, 0.01)))
                               for fl, m in zip(fl_test, preds_test_c)])
        p_nb_cal  = np.array([1.0 - sp_nbinom.cdf(fl, n=nb_r, p=nb_r / (nb_r + max(m, 0.01)))
                               for fl, m in zip(fl_cal, preds_cal_c)])

        # ---- (c) Logística multi-feature ----
        def _logfeat(pred_c, dline, X_df, min_):
            diff = pred_c - dline
            parts = [diff, diff / np.maximum(dline, 1.0), pred_c, dline,
                     np.full(len(pred_c), float(min_))]
            for col in ["game_regime", "residual_corners", "momentum_score",
                        "corners_rate_per_min", "is_high_pace"]:
                if col in X_df.columns:
                    parts.append(X_df[col].fillna(0).values.astype(float))
            return np.column_stack(parts)

        X_lcal  = _logfeat(preds_cal_c,  dynamic_line_cal, X_cal,  snap_min)
        X_ltest = _logfeat(preds_test_c, dynamic_line,     X_test, snap_min)
        if len(np.unique(over_actual_cal)) == 2:
            _log_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            _log_clf.fit(X_lcal, over_actual_cal)
            p_logistic_test = _log_clf.predict_proba(X_ltest)[:, 1]
            p_logistic_cal  = _log_clf.predict_proba(X_lcal)[:, 1]
        else:
            p_logistic_test = p_normal_test.copy()
            p_logistic_cal  = p_normal_cal.copy()

        # ---- (d) XGBoost classificador direto ----
        def _clffeat(pred_c, dline, X_df):
            parts = [pred_c - dline, pred_c, dline]
            for col in ["game_regime", "residual_corners", "momentum_score",
                        "corners_rate_per_min", "corners_total_so_far"]:
                if col in X_df.columns:
                    parts.append(X_df[col].fillna(0).values.astype(float))
            return np.column_stack(parts)

        X_clf_tr = _clffeat(preds_train_c, dynamic_line_train, X_train)
        X_clf_ca = _clffeat(preds_cal_c,   dynamic_line_cal,   X_cal)
        X_clf_te = _clffeat(preds_test_c,  dynamic_line,       X_test)

        if len(np.unique(over_actual_train)) == 2:
            _model_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                eval_metric="logloss", random_state=42, verbosity=0,
                early_stopping_rounds=20,
            )
            _model_clf.fit(X_clf_tr, over_actual_train,
                           eval_set=[(X_clf_ca, over_actual_cal)], verbose=False)
            p_clf_test = _model_clf.predict_proba(X_clf_te)[:, 1]
            p_clf_cal  = _model_clf.predict_proba(X_clf_ca)[:, 1]
        else:
            p_clf_test = p_normal_test.copy()
            p_clf_cal  = p_normal_cal.copy()

        # ---- Seleciona método pelo Brier no CAL set ----
        methods: dict[str, tuple] = {
            "Normal":   (p_normal_test,   p_normal_cal),
            "Poisson":  (p_poisson_test,  p_poisson_cal),
            "NegBinom": (p_nb_test,       p_nb_cal),
            "Logistic": (p_logistic_test, p_logistic_cal),
            "XGBClf":   (p_clf_test,      p_clf_cal),
        }
        brier_cal_scores = {
            name: float(np.mean((p_cal - over_actual_cal) ** 2))
            for name, (_, p_cal) in methods.items()
        }
        best_method = min(brier_cal_scores, key=brier_cal_scores.get)
        p_over     = methods[best_method][0]
        p_over_cal = methods[best_method][1]

        print(f"\n  Método probabilístico (seleção por Brier no cal set):")
        print(f"    {'Método':<12s}  {'Brier(cal)':>10s}")
        for mname, bv in brier_cal_scores.items():
            mark = "  ← SELECIONADO" if mname == best_method else ""
            print(f"    {mname:<12s}  {bv:>10.4f}{mark}")
        if best_method == "NegBinom":
            print(f"    (NegBinom r={nb_r:.2f})")

        # ---- (e) Threshold selecionado no CAL set (sem leakage) ----
        # Over e Under avaliados separadamente; o melhor vence.
        _THRESH_MAX  = 0.68
        _MIN_CAL_BET = 30

        # -- Over: sweep threshold no cal --
        best_over_thresh = BREAKEVEN + MIN_EDGE
        best_over_roi    = -999.0
        for _thr in np.arange(BREAKEVEN + MIN_EDGE, _THRESH_MAX + 0.001, 0.01):
            _mc = p_over_cal >= _thr
            if _mc.sum() < _MIN_CAL_BET:
                continue
            _wc = over_actual_cal[_mc].sum()
            _rc = (_wc * (ODDS_OVER - 1) - (_mc.sum() - _wc)) / _mc.sum()
            if _rc > best_over_roi:
                best_over_roi    = _rc
                best_over_thresh = _thr

        # -- Under: sweep threshold no cal --
        p_under_cal = 1.0 - p_over_cal
        under_actual_cal = 1 - over_actual_cal
        best_under_thresh = BREAKEVEN + MIN_EDGE
        best_under_roi    = -999.0
        for _thr in np.arange(BREAKEVEN + MIN_EDGE, _THRESH_MAX + 0.001, 0.01):
            _mc = p_under_cal >= _thr
            if _mc.sum() < _MIN_CAL_BET:
                continue
            _wc = under_actual_cal[_mc].sum()
            _rc = (_wc * (ODDS_UNDER - 1) - (_mc.sum() - _wc)) / _mc.sum()
            if _rc > best_under_roi:
                best_under_roi    = _rc
                best_under_thresh = _thr

        # Decide a melhor direção (Over vs Under) pelo ROI do cal
        if best_over_roi >= best_under_roi and best_over_roi > 0:
            _bet_side    = "Over"
            best_thresh  = best_over_thresh
            best_roi_on_cal = best_over_roi
        elif best_under_roi > 0:
            _bet_side    = "Under"
            best_thresh  = best_under_thresh
            best_roi_on_cal = best_under_roi
        else:
            # Nenhum lado rentável: usa Over com threshold conservador
            _bet_side    = "Over"
            best_thresh  = BREAKEVEN + MIN_EDGE
            best_roi_on_cal = best_over_roi

        # ---- (h) Odds reais por jogo ----
        odds_over_t  = X_test["corners_over_odds"].values  if "corners_over_odds"  in X_test.columns else None
        odds_under_t = X_test["corners_under_odds"].values if "corners_under_odds" in X_test.columns else None

        def _profit_vec(over_arr, mask, odds_arr, default_odds, is_over=True):
            profit = 0.0
            idxs = np.where(mask)[0]
            for i in idxs:
                try:
                    o = (float(odds_arr[i])
                         if (odds_arr is not None and not np.isnan(float(odds_arr[i]))
                             and float(odds_arr[i]) > 1.0)
                         else default_odds)
                except (TypeError, ValueError):
                    o = default_odds
                won = int(over_arr[i]) if is_over else int(1 - over_arr[i])
                profit += (o - 1) * won - (1 - won)
            n = len(idxs)
            return profit, (profit / n if n else 0.0), n

        # ---- (A) Baseline ----
        n_total  = len(y_test)
        wins_all = over_actual.sum()
        acc_all  = wins_all / n_total
        pf_all   = wins_all * (ODDS_OVER - 1) - (n_total - wins_all)
        roi_all  = pf_all / n_total
        brier_all = float(np.mean((p_over - over_actual) ** 2))

        # ---- Diagnóstico de sharpness: distribuição de P(over) ----
        print(f"\n  Distribuição P(over) [{best_method}]  (sharpness):")
        _bins = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
        _labels = ["<35%","35-40%","40-45%","45-50%","50-55%","55-60%","60-65%","65-70%",">70%"]
        for _lb, _lo, _hi in zip(_labels, _bins[:-1], _bins[1:]):
            _m = (p_over >= _lo) & (p_over < _hi)
            _n = _m.sum()
            _pct = _n / len(p_over)
            _bar = "█" * int(_pct * 40)
            _acc_str = ""
            if _n > 0:
                _acc_str = f"  acur={over_actual[_m].mean():.0%}"
            print(f"    {_lb:>8s}: {_n:>5,} ({_pct:>4.0%}) {_bar}{_acc_str}")
        print(f"    Média={np.mean(p_over):.3f}  Std={np.std(p_over):.3f}  "
              f"Min={np.min(p_over):.3f}  Max={np.max(p_over):.3f}")
        print(f"    > 55%: {(p_over >= 0.55).sum():,}  > 60%: {(p_over >= 0.60).sum():,}  "
              f"> 65%: {(p_over >= 0.65).sum():,}")

        p_under     = 1.0 - p_over
        under_actual = 1 - over_actual

        print(f"\n  Linha por liga  (corners + remaining × league_rate | fallback 0.1/min):")
        print(f"    Linha média test : {dynamic_line.mean():.2f}  "
              f"(min={dynamic_line.min():.1f}  max={dynamic_line.max():.1f})")
        print(f"    Break-even @ {ODDS_OVER:.2f} : {BREAKEVEN:.1%}  |  Edge mín: +{MIN_EDGE:.0%}")
        print(f"    Melhor lado (CAL): {_bet_side}  thresh={best_thresh:.0%}  "
              f"ROI(cal)={best_roi_on_cal:+.1%}")
        print(f"\n    (A) BASELINE — Over SEMPRE:")
        print(f"        Apostas={n_total:,}  Acur={acc_all:.1%}  ROI={roi_all:+.1%}  Brier={brier_all:.4f}")

        # ---- (B) Over — sweep de threshold no test ----
        print(f"\n    (B) Over — [{best_method}] — threshold sweep:")
        print(f"        {'Thresh':>7s}  {'Apostas':>7s}  {'Acur':>6s}  {'ROI(fix)':>9s}  "
              f"{'ROI(real)':>9s}  {'Brier':>7s}")
        _thresh_over_set = sorted({round(float(best_over_thresh), 2)} | {0.50, 0.55, 0.60, 0.65, 0.70})
        for thresh in _thresh_over_set:
            mask_t = p_over >= thresh
            n_t    = mask_t.sum()
            if n_t > 0:
                wins_t      = over_actual[mask_t].sum()
                acc_t       = wins_t / n_t
                roi_fix     = (wins_t * (ODDS_OVER - 1) - (n_t - wins_t)) / n_t
                _, roi_real, _ = _profit_vec(over_actual, mask_t, odds_over_t, ODDS_OVER, True)
                brier_t     = float(np.mean((p_over[mask_t] - over_actual[mask_t]) ** 2))
            else:
                acc_t = roi_fix = roi_real = brier_t = 0.0
            marker = "  ←" if (_bet_side == "Over" and abs(thresh - best_thresh) < 0.005) else ""
            print(f"        {thresh:>6.0%}  {n_t:>7,}  {acc_t:>5.1%}  "
                  f"{roi_fix:>+8.1%}  {roi_real:>+8.1%}  {brier_t:>7.4f}{marker}")

        # ---- (C) Under — sweep de threshold no test ----
        print(f"\n    (C) Under — [{best_method}] — threshold sweep:")
        print(f"        {'Thresh':>7s}  {'Apostas':>7s}  {'Acur':>6s}  {'ROI(fix)':>9s}  "
              f"{'ROI(real)':>9s}")
        _thresh_under_set = sorted({round(float(best_under_thresh), 2)} | {0.50, 0.55, 0.60, 0.65, 0.70})
        for thresh in _thresh_under_set:
            mask_u = p_under >= thresh
            n_u    = mask_u.sum()
            if n_u > 0:
                wins_u  = under_actual[mask_u].sum()
                acc_u   = wins_u / n_u
                roi_fix = (wins_u * (ODDS_UNDER - 1) - (n_u - wins_u)) / n_u
                _, roi_real, _ = _profit_vec(over_actual, mask_u, odds_under_t, ODDS_UNDER, False)
            else:
                acc_u = roi_fix = roi_real = 0.0
            marker = "  ←" if (_bet_side == "Under" and abs(thresh - best_thresh) < 0.005) else ""
            print(f"        {thresh:>6.0%}  {n_u:>7,}  {acc_u:>5.1%}  "
                  f"{roi_fix:>+8.1%}  {roi_real:>+8.1%}{marker}")

        # ---- (i) Breakdown por contexto (lado selecionado) ----
        if _bet_side == "Over":
            mask_best = p_over >= best_thresh
            _act_arr  = over_actual
            _odds_def = ODDS_OVER
        else:
            mask_best = p_under >= best_thresh
            _act_arr  = under_actual
            _odds_def = ODDS_UNDER

        print(f"\n    Breakdown — [{_bet_side}, {best_method}, thresh={best_thresh:.0%}]:")
        print(f"        {'Contexto':<26s}  {'Total':>6s}  {'Apostas':>7s}  {'Acur':>6s}  {'ROI':>8s}")
        _ctx_rows = []
        for _col, _lmap in [
            ("game_regime",   {0: "Regime:LOW", 1: "Regime:NORMAL", 2: "Regime:HIGH"}),
            ("phase_of_game", {0: "Fase:0-30",  1: "Fase:31-60",   2: "Fase:61-75", 3: "Fase:76+"}),
        ]:
            if _col in X_test.columns:
                for _val, _lbl in _lmap.items():
                    _mc = (X_test[_col] == _val).values
                    _mb = mask_best & _mc
                    _nb = _mb.sum()
                    if _nb > 3:
                        _w = _act_arr[_mb].sum()
                        _r = (_w * (_odds_def - 1) - (_nb - _w)) / _nb
                        _ctx_rows.append((_lbl, _mc.sum(), _nb, _w / _nb, _r))
        for _lo, _hi, _lbl in [(0, 8, "Linha:<=8"), (8, 10.5, "Linha:8-10.5"),
                                (10.5, 13, "Linha:10.5-13"), (13, 99, "Linha:>=13")]:
            _ml = (dynamic_line > _lo) & (dynamic_line <= _hi)
            _mb = mask_best & _ml
            _nb = _mb.sum()
            if _nb > 3:
                _w = _act_arr[_mb].sum()
                _r = (_w * (_odds_def - 1) - (_nb - _w)) / _nb
                _ctx_rows.append((_lbl, _ml.sum(), _nb, _w / _nb, _r))
        for _lbl, _ntot, _nb, _ac, _roi in _ctx_rows:
            print(f"        {_lbl:<26s}  {_ntot:>6,}  {_nb:>7,}  {_ac:>5.1%}  {_roi:>+7.1%}")

        # Armazena métricas do lado selecionado
        n_best = mask_best.sum()
        if n_best > 0:
            wins_b       = _act_arr[mask_best].sum()
            accuracy_dyn = wins_b / n_best
            profit_b     = wins_b * (_odds_def - 1) - (n_best - wins_b)
            roi_dyn      = profit_b / n_best
            brier_dyn    = (float(np.mean((p_over[mask_best] - over_actual[mask_best]) ** 2))
                           if _bet_side == "Over" else
                           float(np.mean((p_under[mask_best] - under_actual[mask_best]) ** 2)))
        else:
            accuracy_dyn = acc_all
            roi_dyn      = roi_all
            brier_dyn    = brier_all
            n_best       = n_total

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

        if te_min is not None:
            joblib.dump(te_min, DATA_DIR / f"target_encoder_min{snap_min}.joblib")

        joblib.dump(train_medians, DATA_DIR / f"train_medians_min{snap_min}.joblib")

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
            "dynamic_line_n_bets":   int(n_best),
            "dynamic_line_method":   best_method,
            "dynamic_line_thresh":   best_thresh,
        }

    # --- Salva metadata ---
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
    print(f"\n{'═' * 80}")
    print(f"  LINHA DINÂMICA: corners_atual + (90-min)/10  →  arred. .0/.5")
    print(f"  Odds: 1.83  |  Break-even: {1/1.83:.1%}")
    print(f"  Estratégia: P(over) via método selecionado > threshold")
    print(f"{'═' * 80}")
    print(f"  {'Min':>4s}  {'Linha':>6s}  {'Método':>10s}  {'Thresh':>7s}  "
          f"{'Apostas':>8s}  {'Acurácia':>9s}  {'ROI':>8s}  {'Brier':>7s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}")
    for m, info in all_metadata["models"].items():
        if "dynamic_line_accuracy" not in info:
            continue
        remaining = 90 - m
        linha_formula = f"+{remaining/10:.1f}"
        roi_str    = f"{info['dynamic_line_roi']:+.1%}"
        n_bets     = info.get("dynamic_line_n_bets", "?")
        method_str = info.get("dynamic_line_method", "?")
        thresh_str = f"{info.get('dynamic_line_thresh', 0):.0%}"
        print(f"  {m:>4d}  {linha_formula:>6s}  {method_str:>10s}  {thresh_str:>7s}  "
              f"{str(n_bets):>8s}  {info['dynamic_line_accuracy']:>8.1%}  "
              f"{roi_str:>8s}  {info['dynamic_line_brier']:>7.4f}")
    print(f"{'═' * 80}")

except ImportError as e:
    print(f"  Pacote não instalado ({e}). Instale: pip install xgboost scikit-learn joblib")

# %%
print("\nAnálise concluída!")
