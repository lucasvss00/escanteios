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
    # _home_total = stat do time da casa; _away_total = stat do visitante
    stat_cols = {
        "corners_total":                "corners",
        "corners_home_total":           "corners_scored",
        "corners_away_total":           "corners_conceded",
        "dangerous_attacks_home_total": "dangerous_attacks",
        "dangerous_attacks_away_total": "dangerous_attacks_against",
        "attacks_home_total":           "attacks",
        "attacks_away_total":           "attacks_against",
        "shots_on_target_home_total":   "shots_on_target",
        "shots_on_target_away_total":   "shots_on_target_against",
        "shots_off_target_home_total":  "shots_off_target",
        "shots_off_target_away_total":  "shots_off_target_against",
        "total_goals":                  "goals",
    }
    # Filtra apenas colunas que existem
    stat_cols = {k: v for k, v in stat_cols.items() if k in df.columns}

    # Pares home<->away para swap correto ao registrar histórico do visitante
    _swap_pairs_def = {
        "corners_home_total": "corners_away_total",
        "dangerous_attacks_home_total": "dangerous_attacks_away_total",
        "attacks_home_total": "attacks_away_total",
        "shots_on_target_home_total": "shots_on_target_away_total",
        "shots_off_target_home_total": "shots_off_target_away_total",
    }
    swap_pairs = {k: v for k, v in _swap_pairs_def.items()
                  if k in stat_cols and v in stat_cols}

    # Acumula histórico por time (home_id e away_id)
    team_history: dict[str, list[dict]] = {}
    last_game_date: dict[str, pd.Timestamp] = {}
    result_rows = []

    def _team_feats(hist: list[dict], prefix: str, feat: dict) -> None:
        """Calcula features históricas para um time (home ou away)."""
        for orig_col, alias in stat_cols.items():
            vals = [h.get(orig_col) for h in hist if h.get(orig_col) is not None]
            feat[f"{prefix}_{alias}_avg"] = round(np.mean(vals), 2) if vals else None
        feat[f"{prefix}_games"] = len(hist)

        # Forma recente: últimos 5 / 10 jogos — corners scored
        cs_vals = [h.get("corners_home_total") for h in hist
                   if h.get("corners_home_total") is not None]
        last5 = cs_vals[-5:] if len(cs_vals) >= 5 else cs_vals
        last10 = cs_vals[-10:] if len(cs_vals) >= 10 else cs_vals
        feat[f"{prefix}_corners_last5_avg"] = round(np.mean(last5), 2) if last5 else None
        feat[f"{prefix}_corners_last10_avg"] = round(np.mean(last10), 2) if last10 else None
        # Consistência / volatilidade
        feat[f"{prefix}_corners_std_last10"] = (
            round(float(np.std(last10)), 2) if len(last10) >= 2 else None)

    _null_extras = [
        "corners_last5_avg", "corners_last10_avg", "corners_std_last10",
    ]

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
            _team_feats(hist, "hist_home", feat)
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
            for ex in _null_extras:
                feat[f"hist_home_{ex}"] = None

        # Features históricas do time visitante
        if away_id and away_id in team_history:
            hist = team_history[away_id][-window:]
            _team_feats(hist, "hist_away", feat)
            # Mando: média de escanteios apenas nos jogos FORA
            away_games = [h for h in hist if h.get("_is_home") is False]
            aa_vals = [h.get("corners_home_total") for h in away_games
                       if h.get("corners_home_total") is not None]
            feat["hist_away_corners_away_avg"] = round(np.mean(aa_vals), 2) if aa_vals else None
        else:
            for alias in stat_cols.values():
                feat[f"hist_away_{alias}_avg"] = None
            feat["hist_away_games"] = 0
            feat["hist_away_corners_away_avg"] = None
            for ex in _null_extras:
                feat[f"hist_away_{ex}"] = None

        result_rows.append(feat)

        # Atualiza histórico: time da casa jogou em casa
        game_stats = {col: row.get(col) for col in stat_cols}
        game_stats["_is_home"] = True
        team_history.setdefault(home_id, []).append(game_stats)

        # Para o visitante, inverte TODOS os pares home/away e marca como away
        away_stats = dict(game_stats)
        for home_col, away_col in swap_pairs.items():
            away_stats[home_col] = row.get(away_col)
            away_stats[away_col] = row.get(home_col)
        away_stats["_is_home"] = False
        team_history.setdefault(away_id, []).append(away_stats)

        # Atualiza última data de jogo
        if home_id and pd.notna(current_dt):
            last_game_date[home_id] = current_dt
        if away_id and pd.notna(current_dt):
            last_game_date[away_id] = current_dt

    return pd.DataFrame(result_rows)


def build_h2h_history(df_pano: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada jogo, calcula features de confronto direto (head-to-head)
    com base APENAS nos jogos anteriores entre os dois mesmos times.
    Sem data leakage: o histórico do par é atualizado APÓS calcular as features.

    Retorna DataFrame com event_id + features h2h_*.
    """
    if "kickoff_dt" not in df_pano.columns:
        return pd.DataFrame({"event_id": df_pano["event_id"]})

    df = df_pano.copy()
    df["kickoff_dt_parsed"] = pd.to_datetime(df["kickoff_dt"], errors="coerce")
    df = df.sort_values("kickoff_dt_parsed").reset_index(drop=True)

    # Histórico por par ordenado (simétrico): chave = frozenset({home_id, away_id})
    # Cada item guarda: {"date", "corners_total", "home_id", "corners_home", "corners_away"}
    pair_history: dict[frozenset, list[dict]] = {}
    result_rows: list[dict] = []

    for _, row in df.iterrows():
        event_id = row["event_id"]
        home_id = str(row.get("home_id", ""))
        away_id = str(row.get("away_id", ""))
        current_dt = row.get("kickoff_dt_parsed")

        feat: dict = {"event_id": event_id}

        if home_id and away_id and home_id != away_id:
            key = frozenset({home_id, away_id})
            hist = pair_history.get(key, [])
        else:
            hist = []

        if hist:
            totals = [h["corners_total"] for h in hist if h.get("corners_total") is not None]
            feat["h2h_games"] = len(hist)
            feat["h2h_corners_avg"] = round(float(np.mean(totals)), 2) if totals else None
            feat["h2h_corners_std"] = (round(float(np.std(totals)), 2)
                                        if len(totals) >= 2 else None)

            # Últimos 3 confrontos
            last3 = totals[-3:] if len(totals) >= 1 else []
            feat["h2h_last3_avg"] = round(float(np.mean(last3)), 2) if last3 else None

            # Média de escanteios do time atualmente em casa quando ELE jogou em casa
            # contra este adversário (mesma perspectiva)
            home_as_home = [h["corners_home"] for h in hist
                            if h.get("home_id") == home_id and h.get("corners_home") is not None]
            feat["h2h_corners_home_avg"] = (round(float(np.mean(home_as_home)), 2)
                                             if home_as_home else None)

            # Média de escanteios do visitante atual quando ELE jogou fora contra este adversário
            away_as_away = [h["corners_away"] for h in hist
                            if h.get("home_id") == home_id and h.get("corners_away") is not None]
            feat["h2h_corners_away_avg"] = (round(float(np.mean(away_as_away)), 2)
                                             if away_as_away else None)

            last_dt = hist[-1].get("date")
            if pd.notna(current_dt) and pd.notna(last_dt):
                feat["h2h_days_since_last"] = int((current_dt - last_dt).days)
            else:
                feat["h2h_days_since_last"] = None
        else:
            feat["h2h_games"] = 0
            feat["h2h_corners_avg"] = None
            feat["h2h_corners_std"] = None
            feat["h2h_last3_avg"] = None
            feat["h2h_corners_home_avg"] = None
            feat["h2h_corners_away_avg"] = None
            feat["h2h_days_since_last"] = None

        result_rows.append(feat)

        # Atualiza APÓS calcular (sem leakage)
        if home_id and away_id and home_id != away_id:
            key = frozenset({home_id, away_id})
            ct = row.get("corners_total")
            ch = row.get("corners_home_total")
            ca = row.get("corners_away_total")
            pair_history.setdefault(key, []).append({
                "date": current_dt,
                "corners_total": float(ct) if ct is not None and pd.notna(ct) else None,
                "corners_home": float(ch) if ch is not None and pd.notna(ch) else None,
                "corners_away": float(ca) if ca is not None and pd.notna(ca) else None,
                "home_id": home_id,  # quem foi mandante NESTE confronto passado
            })

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

print("Calculando histórico de confrontos diretos (H2H)...")
df_h2h = build_h2h_history(df_pano)
_n_with_h2h = int((df_h2h["h2h_games"] > 0).sum()) if "h2h_games" in df_h2h.columns else 0
print(f"H2H calculado para {len(df_h2h):,} jogos ({_n_with_h2h:,} com histórico prévio)")

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
                         league_std_map: dict | None = None,
                         df_h2h: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Versão vetorizada — ~10-50× mais rápida que a versão com loop.
    Para cada jogo e cada minuto de snapshot:
      - Features: tudo que aconteceu ATÉ aquele minuto
      - Features pré-jogo: odds, H2H, histórico dos times
      - Features temporais: dia da semana, hora, mês
      - Target: total de escanteios do jogo inteiro (de df_pano)
    """
    import logging
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 0. Preparação: filtrar jogos válidos, criar cross-join
    # ------------------------------------------------------------------
    valid_events = set(df_pano["event_id"].unique()) & set(df_snap["event_id"].unique())
    snap = df_snap[df_snap["event_id"].isin(valid_events)].copy()
    snap = snap.sort_values(["event_id", "minute"]).reset_index(drop=True)

    # Colunas numéricas do snapshot — preencher NaN com 0 para aritmética
    _STAT_COLS = [
        "corners_home", "corners_away",
        "attacks_home", "attacks_away",
        "dangerous_attacks_home", "dangerous_attacks_away",
        "shots_on_target_home", "shots_on_target_away",
        "shots_off_target_home", "shots_off_target_away",
        "yellow_cards_home", "yellow_cards_away",
        "red_cards_home", "red_cards_away",
        "fouls_home", "fouls_away",
        "saves_home", "saves_away",
        "offsides_home", "offsides_away",
        "goal_kicks_home", "goal_kicks_away",
        "score_home", "score_away",
    ]
    _STAT_COLS = [c for c in _STAT_COLS if c in snap.columns]
    _POSS_COLS = [c for c in ["possession_home", "possession_away"] if c in snap.columns]

    # ------------------------------------------------------------------
    # 1. Para cada snapshot minute, pegar o "last" row (maior minute <= snap_min)
    #    e os valores em (snap_min - N) para janelas de N minutos
    # ------------------------------------------------------------------
    all_frames = []

    for snap_min in snapshot_minutes:
        log.info(f"  build_live_features: processando minuto {snap_min}...")

        until = snap[snap["minute"] <= snap_min].copy()
        if until.empty:
            continue

        # "last" = última linha por event_id (minute mais alto <= snap_min)
        last = until.groupby("event_id").last().reset_index()
        # n_snap_minutes = contagem de linhas por event_id
        n_mins = until.groupby("event_id").size().reset_index(name="n_snap_minutes")
        last = last.merge(n_mins, on="event_id", how="left")
        last["snap_minute"] = snap_min

        # Possession mean até snap_min
        for pc in _POSS_COLS:
            _pmean = until.groupby("event_id")[pc].mean().reset_index(name=f"{pc}_avg_tmp")
            last = last.merge(_pmean, on="event_id", how="left")

        # --- Windowed values: valor no minuto (snap_min - N) ---
        def _get_past_values(n: int, cols: list[str]) -> pd.DataFrame:
            """Pega o último valor de cada col no minuto <= (snap_min - n)."""
            past = snap[snap["minute"] <= (snap_min - n)]
            if past.empty:
                return pd.DataFrame({"event_id": last["event_id"]})
            past_last = past.groupby("event_id")[cols].last().reset_index()
            return past_last

        _window_cols = [c for c in [
            "corners_home", "corners_away",
            "attacks_home", "attacks_away",
            "dangerous_attacks_home", "dangerous_attacks_away",
            "shots_on_target_home", "shots_on_target_away",
            "shots_off_target_home", "shots_off_target_away",
        ] if c in snap.columns]

        past5 = _get_past_values(5, _window_cols)
        past10 = _get_past_values(10, _window_cols)
        past15 = _get_past_values(15, _window_cols)

        # Rename past columns
        for df_past, suffix in [(past5, "_p5"), (past10, "_p10"), (past15, "_p15")]:
            for c in df_past.columns:
                if c != "event_id":
                    df_past.rename(columns={c: c + suffix}, inplace=True)

        last = last.merge(past5, on="event_id", how="left")
        last = last.merge(past10, on="event_id", how="left")
        last = last.merge(past15, on="event_id", how="left")

        # --- First corner minute per event ---
        _ct = until.copy()
        _ct["_ct"] = _ct["corners_home"].fillna(0) + _ct["corners_away"].fillna(0)
        _first_corner = _ct[_ct["_ct"] > 0].groupby("event_id")["minute"].first().reset_index(
            name="_first_corner_min")
        last = last.merge(_first_corner, on="event_id", how="left")

        # --- Half-time values (corners at minute 45) ---
        if snap_min > 45:
            at45 = snap[snap["minute"] <= 45]
            if not at45.empty:
                ht_vals = at45.groupby("event_id")[["corners_home", "corners_away"]].last().reset_index()
                ht_vals.rename(columns={"corners_home": "_ht_ch", "corners_away": "_ht_ca"}, inplace=True)
                last = last.merge(ht_vals, on="event_id", how="left")
            else:
                last["_ht_ch"] = 0
                last["_ht_ca"] = 0
        else:
            last["_ht_ch"] = np.nan
            last["_ht_ca"] = np.nan

        # --- Time since last corner (vectorized) ---
        _ct2 = until.copy()
        _ct2["_total_c"] = _ct2["corners_home"].fillna(0) + _ct2["corners_away"].fillna(0)
        _ct2["_diff_c"] = _ct2.groupby("event_id")["_total_c"].diff()
        _corner_events = _ct2[_ct2["_diff_c"] > 0]
        _last_corner_min = _corner_events.groupby("event_id")["minute"].last().reset_index(
            name="_last_corner_minute")
        last = last.merge(_last_corner_min, on="event_id", how="left")

        # Time since last shot
        _ct2["_total_s"] = _ct2.get("shots_on_target_home", pd.Series(0, index=_ct2.index)).fillna(0) + \
                           _ct2.get("shots_on_target_away", pd.Series(0, index=_ct2.index)).fillna(0)
        _ct2["_diff_s"] = _ct2.groupby("event_id")["_total_s"].diff()
        _shot_events = _ct2[_ct2["_diff_s"] > 0]
        _last_shot_min = _shot_events.groupby("event_id")["minute"].last().reset_index(
            name="_last_shot_minute")
        last = last.merge(_last_shot_min, on="event_id", how="left")

        # Time since last dangerous attack
        _ct2["_total_da"] = _ct2.get("dangerous_attacks_home", pd.Series(0, index=_ct2.index)).fillna(0) + \
                            _ct2.get("dangerous_attacks_away", pd.Series(0, index=_ct2.index)).fillna(0)
        _ct2["_diff_da"] = _ct2.groupby("event_id")["_total_da"].diff()
        _da_events = _ct2[_ct2["_diff_da"] > 0]
        _last_da_min = _da_events.groupby("event_id")["minute"].last().reset_index(
            name="_last_da_minute")
        last = last.merge(_last_da_min, on="event_id", how="left")

        all_frames.append(last)

    # ------------------------------------------------------------------
    # 2. Concatenar todos os snapshots e computar features vetorialmente
    # ------------------------------------------------------------------
    df = pd.concat(all_frames, ignore_index=True)
    SM = df["snap_minute"]

    # Preencher NaN de stats com 0 para aritmética segura
    for c in _STAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # --- Colunas base ---
    df["corners_home_so_far"] = df["corners_home"]
    df["corners_away_so_far"] = df["corners_away"]
    c_home = df["corners_home"].values
    c_away = df["corners_away"].values
    c_total = c_home + c_away
    df["corners_total_so_far"] = c_total

    df["corners_rate_per_min"] = np.round(c_total / np.maximum(SM, 1), 4)

    # Posse média
    for pc in _POSS_COLS:
        avg_col = f"{pc}_avg_tmp"
        target_col = f"{pc[:-len('_home') if pc.endswith('_home') else -len('_away')]}_avg" if False else pc.replace("possession_", "possession_") + "_avg"
        # Simplify: possession_home -> possession_home_avg
        target_col = pc + "_avg"
        if avg_col in df.columns:
            df[target_col] = df[avg_col].round(2)
            df.drop(columns=[avg_col], inplace=True)
        else:
            df[target_col] = np.nan

    # Ataques / chutes / totais
    att_h = df.get("attacks_home", pd.Series(0, index=df.index)).fillna(0).values
    att_a = df.get("attacks_away", pd.Series(0, index=df.index)).fillna(0).values
    da_h = df.get("dangerous_attacks_home", pd.Series(0, index=df.index)).fillna(0).values
    da_a = df.get("dangerous_attacks_away", pd.Series(0, index=df.index)).fillna(0).values
    son_h = df.get("shots_on_target_home", pd.Series(0, index=df.index)).fillna(0).values
    son_a = df.get("shots_on_target_away", pd.Series(0, index=df.index)).fillna(0).values
    soff_h = df.get("shots_off_target_home", pd.Series(0, index=df.index)).fillna(0).values
    soff_a = df.get("shots_off_target_away", pd.Series(0, index=df.index)).fillna(0).values
    yc_h = df.get("yellow_cards_home", pd.Series(0, index=df.index)).fillna(0).values
    yc_a = df.get("yellow_cards_away", pd.Series(0, index=df.index)).fillna(0).values
    rc_h = df.get("red_cards_home", pd.Series(0, index=df.index)).fillna(0).values
    rc_a = df.get("red_cards_away", pd.Series(0, index=df.index)).fillna(0).values
    fouls_h = df.get("fouls_home", pd.Series(0, index=df.index)).fillna(0).values
    fouls_a = df.get("fouls_away", pd.Series(0, index=df.index)).fillna(0).values
    saves_h = df.get("saves_home", pd.Series(0, index=df.index)).fillna(0).values
    saves_a = df.get("saves_away", pd.Series(0, index=df.index)).fillna(0).values
    offs_h = df.get("offsides_home", pd.Series(0, index=df.index)).fillna(0).values
    offs_a = df.get("offsides_away", pd.Series(0, index=df.index)).fillna(0).values
    gk_h = df.get("goal_kicks_home", pd.Series(0, index=df.index)).fillna(0).values
    gk_a = df.get("goal_kicks_away", pd.Series(0, index=df.index)).fillna(0).values
    score_h = df.get("score_home", pd.Series(0, index=df.index)).fillna(0).values
    score_a = df.get("score_away", pd.Series(0, index=df.index)).fillna(0).values

    total_attacks = att_h + att_a
    total_dangerous = da_h + da_a
    total_shots = son_h + son_a + soff_h + soff_a
    shots_on = son_h + son_a

    snap_min_v = SM.values.astype(float)

    # --- Contexto placar ---
    df["score_diff"] = score_h - score_a
    df["total_red_cards"] = rc_h + rc_a
    df["red_card_diff"] = rc_h - rc_a

    # --- Diferenças home - away ---
    df["corners_diff"] = c_home - c_away
    df["attacks_diff"] = att_h - att_a
    df["dangerous_attacks_diff"] = da_h - da_a
    df["shots_on_target_diff"] = son_h - son_a

    # --- Taxas ---
    df["dangerous_attacks_rate"] = np.round((da_h + da_a) / np.maximum(snap_min_v, 1), 4)
    df["corners_per_attack_ratio"] = np.round(c_total / np.maximum(total_attacks, 1), 4)

    # --- Windowed features ---
    def _windowed(col, suffix):
        """Valor current - valor no passado = incremento nos últimos N min."""
        now_v = df.get(col, pd.Series(0, index=df.index)).fillna(0).values
        past_col = f"{col}{suffix}"
        past_v = df.get(past_col, pd.Series(np.nan, index=df.index)).fillna(np.nan).values
        result = np.where(np.isnan(past_v), np.nan, now_v - past_v)
        return result

    # Corners last 5/10/15
    cl5h = _windowed("corners_home", "_p5")
    cl5a = _windowed("corners_away", "_p5")
    cl10h = _windowed("corners_home", "_p10")
    cl10a = _windowed("corners_away", "_p10")
    cl15h = _windowed("corners_home", "_p15")
    cl15a = _windowed("corners_away", "_p15")

    corners_last_5 = np.nan_to_num(cl5h, 0) + np.nan_to_num(cl5a, 0)
    corners_last_10 = np.nan_to_num(cl10h, 0) + np.nan_to_num(cl10a, 0)
    corners_last_15 = np.nan_to_num(cl15h, 0) + np.nan_to_num(cl15a, 0)

    df["corners_last_5min"] = corners_last_5
    df["corners_last_10min"] = corners_last_10
    df["corners_acceleration_5_10"] = corners_last_5 - corners_last_10
    df["corners_rate_last_5"] = np.round(corners_last_5 / 5.0, 4)
    df["corners_rate_last_10"] = np.round(corners_last_10 / 10.0, 4)

    df["corners_last_15_home"] = np.where(np.isnan(cl15h), None, cl15h)
    df["corners_last_15_away"] = np.where(np.isnan(cl15a), None, cl15a)

    df["home_corners_last_5"] = np.where(np.isnan(cl5h), None, cl5h)
    df["away_corners_last_5"] = np.where(np.isnan(cl5a), None, cl5a)
    df["home_corners_last_10"] = np.where(np.isnan(cl10h), None, cl10h)
    df["away_corners_last_10"] = np.where(np.isnan(cl10a), None, cl10a)

    # Attacks, dangerous attacks, shots last 5/10
    att_l5h = np.nan_to_num(_windowed("attacks_home", "_p5"), 0)
    att_l5a = np.nan_to_num(_windowed("attacks_away", "_p5"), 0)
    att_l10h = np.nan_to_num(_windowed("attacks_home", "_p10"), 0)
    att_l10a = np.nan_to_num(_windowed("attacks_away", "_p10"), 0)
    da_l5h = np.nan_to_num(_windowed("dangerous_attacks_home", "_p5"), 0)
    da_l5a = np.nan_to_num(_windowed("dangerous_attacks_away", "_p5"), 0)
    da_l10h = np.nan_to_num(_windowed("dangerous_attacks_home", "_p10"), 0)
    da_l10a = np.nan_to_num(_windowed("dangerous_attacks_away", "_p10"), 0)
    son_l5h = np.nan_to_num(_windowed("shots_on_target_home", "_p5"), 0)
    son_l5a = np.nan_to_num(_windowed("shots_on_target_away", "_p5"), 0)
    soff_l5h = np.nan_to_num(_windowed("shots_off_target_home", "_p5"), 0)
    soff_l5a = np.nan_to_num(_windowed("shots_off_target_away", "_p5"), 0)

    att_last5 = att_l5h + att_l5a
    att_last10 = att_l10h + att_l10a
    da_last5 = da_l5h + da_l5a
    da_last10 = da_l10h + da_l10a
    shots_last5 = son_l5h + son_l5a + soff_l5h + soff_l5a

    df["attacks_last_5min"] = att_last5
    df["attacks_last_10min"] = att_last10
    df["dangerous_attacks_last_5min"] = da_last5
    df["dangerous_attacks_last_10min"] = da_last10
    df["shots_last_5min"] = shots_last5

    # 15 min windows for momentum score
    att_l15h = np.nan_to_num(_windowed("attacks_home", "_p15"), 0)
    att_l15a = np.nan_to_num(_windowed("attacks_away", "_p15"), 0)
    da_l15h = np.nan_to_num(_windowed("dangerous_attacks_home", "_p15"), 0)
    da_l15a = np.nan_to_num(_windowed("dangerous_attacks_away", "_p15"), 0)

    # --- Estado do jogo ---
    is_draw = (score_h == score_a).astype(int)
    df["is_draw"] = is_draw
    is_home_winning = score_h > score_a
    is_away_winning = score_a > score_h

    # Leading/losing team pressure (corners last 10)
    h10 = np.nan_to_num(cl10h, 0)
    a10 = np.nan_to_num(cl10a, 0)
    df["leading_team_pressure"] = np.where(is_home_winning, h10,
                                   np.where(is_away_winning, a10, 0))
    df["losing_team_pressure"] = np.where(is_home_winning, a10,
                                  np.where(is_away_winning, h10, 0))

    # Tempo restante
    df["time_remaining"] = 90 - snap_min_v
    df["total_time_remaining"] = 90 - snap_min_v + 3

    # Proporção e taxa por time
    c_total_safe = np.maximum(c_total, 1)
    df["home_corner_share"] = np.round(c_home / c_total_safe, 4)
    df["away_corner_share"] = np.round(c_away / c_total_safe, 4)
    df["home_corners_rate"] = np.round(c_home / np.maximum(snap_min_v, 1), 4)
    df["away_corners_rate"] = np.round(c_away / np.maximum(snap_min_v, 1), 4)

    # --- Merge com histórico ---
    hist_cols = [c for c in df_hist.columns if c.startswith("hist_") or c in
                 ["event_id", "days_rest_home", "days_rest_away"]]
    if "event_id" not in hist_cols:
        hist_cols = ["event_id"] + hist_cols
    df = df.merge(df_hist[hist_cols], on="event_id", how="left")

    # --- Merge com H2H (confronto direto) ---
    if df_h2h is not None and len(df_h2h) > 0:
        h2h_cols = [c for c in df_h2h.columns if c.startswith("h2h_") or c == "event_id"]
        df = df.merge(df_h2h[h2h_cols], on="event_id", how="left")

    # Expectativa combinada (hist)
    haf = df.get("hist_home_corners_scored_avg", pd.Series(np.nan, index=df.index)).astype(float)
    haa = df.get("hist_home_corners_conceded_avg", pd.Series(np.nan, index=df.index)).astype(float)
    aaf = df.get("hist_away_corners_scored_avg", pd.Series(np.nan, index=df.index)).astype(float)
    aaa = df.get("hist_away_corners_conceded_avg", pd.Series(np.nan, index=df.index)).astype(float)

    home_exp = (haf + aaa) / 2
    away_exp = (aaf + haa) / 2
    match_exp = home_exp + away_exp
    # NaN where any component was NaN
    home_exp = np.where(haf.isna() | aaa.isna(), np.nan, home_exp)
    away_exp = np.where(aaf.isna() | haa.isna(), np.nan, away_exp)
    match_exp = np.where(np.isnan(home_exp) | np.isnan(away_exp), np.nan, match_exp)

    df["home_expected_corners"] = np.round(home_exp, 4)
    df["away_expected_corners"] = np.round(away_exp, 4)
    df["match_expected_corners"] = np.round(match_exp, 4)

    # Desvio do ritmo esperado
    exp_at_min = match_exp * snap_min_v / 90
    df["corners_vs_expected"] = np.where(
        np.isnan(match_exp) | (match_exp <= 0), np.nan,
        np.round(c_total - exp_at_min, 4))
    df["pace_ratio"] = np.where(
        np.isnan(match_exp) | (match_exp <= 0), np.nan,
        np.round(c_total / np.maximum(exp_at_min, 0.1), 4))

    # --- Sinal precoce ---
    hist_combo_avg = (haf + aaf) / 2
    hist_combo_avg = np.where(haf.isna() | aaf.isna(), np.nan, hist_combo_avg)
    df["hist_team_combo_avg"] = np.round(hist_combo_avg, 4)

    # League avg map
    df["_league_avg"] = df["event_id"].astype(str).map(league_avg_map or {})
    df["_league_std"] = df["event_id"].astype(str).map(league_std_map or {})
    league_avg_v = df["_league_avg"].values.astype(float)
    league_std_v = df["_league_std"].values.astype(float)

    df["league_avg_corners"] = df["_league_avg"]
    df["league_std_corners"] = df["_league_std"]

    df["hist_vs_league"] = np.where(
        np.isnan(hist_combo_avg) | np.isnan(league_avg_v), np.nan,
        np.round(hist_combo_avg - league_avg_v / 2, 4))

    hist_def = (haa + aaa) / 2
    hist_def = np.where(haa.isna() | aaa.isna(), np.nan, hist_def)
    df["hist_defensive_strength"] = np.round(hist_def, 4)

    # ------------------------------------------------------------------
    # Features históricas pré-jogo (normalizadas pela liga)
    # ------------------------------------------------------------------
    _la = np.where((league_avg_v <= 0) | np.isnan(league_avg_v), np.nan, league_avg_v)
    _la_safe = np.maximum(np.nan_to_num(_la, nan=1.0), 0.01)

    # Força de ataque / fraqueza defensiva (normalizada pela liga)
    df["home_attack_strength"] = np.round(np.where(
        np.isnan(_la) | haf.isna(), np.nan, haf.values / _la_safe), 4)
    df["away_attack_strength"] = np.round(np.where(
        np.isnan(_la) | aaf.isna(), np.nan, aaf.values / _la_safe), 4)
    df["home_defense_weakness"] = np.round(np.where(
        np.isnan(_la) | haa.isna(), np.nan, haa.values / _la_safe), 4)
    df["away_defense_weakness"] = np.round(np.where(
        np.isnan(_la) | aaa.isna(), np.nan, aaa.values / _la_safe), 4)

    # Matchup direto (interação ataque vs defesa)
    _has = df["home_attack_strength"].values
    _aas = df["away_attack_strength"].values
    _hdw = df["home_defense_weakness"].values
    _adw = df["away_defense_weakness"].values
    df["expected_corners_home"] = np.round(_has * _adw, 4)
    df["expected_corners_away"] = np.round(_aas * _hdw, 4)
    df["expected_corners_match"] = np.round(_has * _adw + _aas * _hdw, 4)

    # Forma recente: últimos 5/10 jogos
    df["home_corners_last5_avg"] = df.get(
        "hist_home_corners_last5_avg", pd.Series(np.nan, index=df.index)).astype(float)
    df["away_corners_last5_avg"] = df.get(
        "hist_away_corners_last5_avg", pd.Series(np.nan, index=df.index)).astype(float)
    df["home_corners_last10_avg"] = df.get(
        "hist_home_corners_last10_avg", pd.Series(np.nan, index=df.index)).astype(float)
    df["away_corners_last10_avg"] = df.get(
        "hist_away_corners_last10_avg", pd.Series(np.nan, index=df.index)).astype(float)

    # Consistência / volatilidade
    df["home_corners_std_last10"] = df.get(
        "hist_home_corners_std_last10", pd.Series(np.nan, index=df.index)).astype(float)
    df["away_corners_std_last10"] = df.get(
        "hist_away_corners_std_last10", pd.Series(np.nan, index=df.index)).astype(float)

    # Ajuste por adversário (força do calendário)
    # Usa corners_conceded do oponente atual como proxy de schedule strength
    _hcc_safe = np.maximum(np.nan_to_num(haa.values, nan=1.0), 0.01)
    _acc_safe = np.maximum(np.nan_to_num(aaa.values, nan=1.0), 0.01)
    df["home_adjusted_corners"] = np.round(np.where(
        haf.isna() | aaa.isna(), np.nan, haf.values / _acc_safe), 4)
    df["away_adjusted_corners"] = np.round(np.where(
        aaf.isna() | haa.isna(), np.nan, aaf.values / _hcc_safe), 4)

    # Estilo de jogo (proxy): corners por chute e por ataque perigoso
    hist_son_h = df.get("hist_home_shots_on_target_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_soff_h = df.get("hist_home_shots_off_target_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_son_a = df.get("hist_away_shots_on_target_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_soff_a = df.get("hist_away_shots_off_target_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_shots_h = hist_son_h.values + np.nan_to_num(hist_soff_h.values, nan=0.0)
    hist_shots_a = hist_son_a.values + np.nan_to_num(hist_soff_a.values, nan=0.0)

    hist_da_h_v = df.get("hist_home_dangerous_attacks_avg", pd.Series(np.nan, index=df.index)).astype(float).values
    hist_da_a_v = df.get("hist_away_dangerous_attacks_avg", pd.Series(np.nan, index=df.index)).astype(float).values

    df["home_corners_per_shot"] = np.round(np.where(
        haf.isna() | (hist_shots_h <= 0), np.nan,
        haf.values / np.maximum(hist_shots_h, 0.01)), 4)
    df["away_corners_per_shot"] = np.round(np.where(
        aaf.isna() | (hist_shots_a <= 0), np.nan,
        aaf.values / np.maximum(hist_shots_a, 0.01)), 4)
    df["home_corners_per_dangerous"] = np.round(np.where(
        haf.isna() | np.isnan(hist_da_h_v) | (hist_da_h_v <= 0), np.nan,
        haf.values / np.maximum(hist_da_h_v, 0.01)), 4)
    df["away_corners_per_dangerous"] = np.round(np.where(
        aaf.isna() | np.isnan(hist_da_a_v) | (hist_da_a_v <= 0), np.nan,
        aaf.values / np.maximum(hist_da_a_v, 0.01)), 4)

    # Desvio em relação à liga
    df["home_vs_league"] = np.round(np.where(
        haf.isna() | np.isnan(_la), np.nan, haf.values - _la / 2), 4)
    df["away_vs_league"] = np.round(np.where(
        aaf.isna() | np.isnan(_la), np.nan, aaf.values - _la / 2), 4)

    # Prior de jogo (baseline pré-live)
    df["pre_match_expected_total"] = np.round(np.where(
        np.isnan(_la) | np.isnan(_has * _adw + _aas * _hdw), np.nan,
        (_has * _adw + _aas * _hdw) * _la), 4)

    df["no_corner_yet"] = (c_total == 0).astype(int)
    df["early_corner_surge"] = np.round(c_total / np.maximum(snap_min_v / 15, 1), 4)

    # First corner speed
    _fcm = df.get("_first_corner_min", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["first_corner_speed"] = np.where(
        np.isnan(_fcm) | (_fcm <= 0), 0,
        np.round(1.0 / np.maximum(_fcm, 1), 4))

    # hist_actual_rate_ratio
    exp_rate_pm = match_exp / 90
    actual_rate = c_total / np.maximum(snap_min_v, 1)
    df["hist_actual_rate_ratio"] = np.where(
        np.isnan(match_exp) | (match_exp <= 0), np.nan,
        np.round(actual_rate / np.maximum(exp_rate_pm, 0.01), 4))

    # hist_combined_dangerous
    hist_da_h = df.get("hist_home_dangerous_attacks_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_da_a = df.get("hist_away_dangerous_attacks_avg", pd.Series(np.nan, index=df.index)).astype(float)
    hist_da_combo = (hist_da_h + hist_da_a) / 2
    df["hist_combined_dangerous"] = np.where(
        hist_da_h.isna() | hist_da_a.isna(), np.nan,
        np.round(hist_da_combo, 4))

    # --- Liga: z-score ---
    exp_at_min_lg = league_avg_v * snap_min_v / 90
    df["z_score_corners"] = np.where(
        np.isnan(league_avg_v) | np.isnan(league_std_v) | (league_std_v <= 0), np.nan,
        np.round((c_total - exp_at_min_lg) / league_std_v, 4))

    # Team style
    half_league = league_avg_v / 2
    df["team_style_home"] = np.where(
        np.isnan(league_avg_v) | haf.isna(), np.nan,
        np.round(haf.values - half_league, 4))
    df["team_style_away"] = np.where(
        np.isnan(league_avg_v) | aaf.isna(), np.nan,
        np.round(aaf.values - half_league, 4))

    # Intensity index
    league_rate = league_avg_v / 90
    df["intensity_index"] = np.where(
        np.isnan(league_avg_v) | (league_avg_v <= 0), np.nan,
        np.round(df["corners_rate_last_10"].values / np.maximum(league_rate, 0.01), 4))

    # Late game boost
    df["late_game_boost"] = np.round(corners_last_10 * (snap_min_v / 90), 4)

    # Pressure index per team (last 10)
    df["pressure_index_home"] = np.round(h10 / np.maximum(a10, 0.5), 4)
    df["pressure_index_away"] = np.round(a10 / np.maximum(h10, 0.5), 4)

    # Game state
    df["game_state_factor"] = np.round((score_h - score_a) * snap_min_v, 4)
    df["comeback_pressure"] = ((snap_min_v > 60) & (score_h != score_a)).astype(int)
    df["dominance_index"] = np.round(np.abs(df["home_corner_share"].values - 0.5), 4)
    df["volatility_index"] = np.round(np.abs(
        df["corners_rate_last_5"].values - df["corners_rate_last_10"].values), 4)
    df["momentum_shift"] = np.round(corners_last_5 - corners_last_10 / 2, 4)

    # Remaining projections
    total_time_rem = df["total_time_remaining"].values
    expected_remaining = df["corners_rate_per_min"].values * total_time_rem
    adjusted_expected_remaining = df["corners_rate_last_10"].values * total_time_rem

    df["expected_remaining_corners"] = np.round(expected_remaining, 4)
    df["adjusted_expected_remaining"] = np.round(adjusted_expected_remaining, 4)

    # Ratio: actual remaining / expected remaining
    ct_final_tmp = df.get("corners_total", pd.Series(np.nan, index=df.index)).values
    actual_remaining = ct_final_tmp - c_total
    df["remaining_vs_expected"] = np.round(np.where(
        np.isnan(ct_final_tmp) | (expected_remaining < 0.01),
        np.nan,
        actual_remaining / np.maximum(expected_remaining, 0.01)), 4)

    # --- 1st vs 2nd half analysis ---
    ht_ch = df["_ht_ch"].fillna(0).values
    ht_ca = df["_ht_ca"].fillna(0).values
    first_half = ht_ch + ht_ca
    second_half_so_far = c_total - first_half
    mins_2nd = np.maximum(snap_min_v - 45, 1)
    second_half_rate = second_half_so_far / mins_2nd
    first_half_rate = first_half / 45

    is_2nd_half = snap_min_v > 45
    df["first_half_corners"] = np.where(is_2nd_half, first_half, np.nan)
    df["second_half_corners_so_far"] = np.where(is_2nd_half, second_half_so_far, np.nan)
    df["second_half_rate"] = np.where(is_2nd_half, np.round(second_half_rate, 4), np.nan)
    df["delta_rate_halves"] = np.where(is_2nd_half,
        np.round(second_half_rate - first_half_rate, 4), np.nan)
    df["fatigue_factor"] = np.where(
        is_2nd_half & (first_half > 0),
        np.round(df["corners_rate_last_10"].values / np.maximum(first_half_rate, 0.01), 4),
        np.nan)

    # --- Additional features ---
    poss_h = df.get("possession_home_avg", pd.Series(np.nan, index=df.index)).values.astype(float)
    poss_a = df.get("possession_away_avg", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["possession_diff"] = np.where(np.isnan(poss_h) | np.isnan(poss_a), np.nan, poss_h - poss_a)
    df["fouls_diff"] = fouls_h - fouls_a
    df["fouls_total"] = fouls_h + fouls_a
    df["corners_per_dangerous_attack"] = np.round(c_total / np.maximum(total_dangerous, 1), 4)
    df["shots_per_corner"] = np.round(total_shots / np.maximum(c_total, 1), 4)

    # Totals
    df["saves_total"] = saves_h + saves_a
    df["shots_total"] = total_shots
    df["offsides_total"] = offs_h + offs_a
    df["cards_total"] = yc_h + yc_a + rc_h + rc_a
    df["goal_kicks_total"] = gk_h + gk_a
    df["corners_per_goal_kick"] = np.round(c_total / np.maximum(gk_h + gk_a, 1), 4)

    # --- Merge panorama data (pre-game odds, HT score, targets) ---
    pano_cols = ["event_id", "corners_total",
                 "corners_home_total", "corners_away_total",
                 "ht_score_home", "ht_score_away",
                 "corners_line", "corners_over_odds", "corners_under_odds",
                 "asian_corners_line", "asian_corners_home_odds", "asian_corners_away_odds",
                 "odds_home_win", "odds_draw", "odds_away_win",
                 "goals_line", "goals_over_odds", "goals_under_odds",
                 "btts_yes_odds", "btts_no_odds",
                 "live_corners_line", "live_corners_over_odds", "live_corners_under_odds",
                 "throw_ins_home_total", "throw_ins_away_total",
                 "tackles_home_total", "tackles_away_total"]
    pano_cols = [c for c in pano_cols if c in df_pano.columns]
    df = df.merge(df_pano[pano_cols].drop_duplicates("event_id"), on="event_id", how="left")

    # HT score only for snap > 45
    if "ht_score_home" in df.columns:
        df["ht_score_home"] = np.where(is_2nd_half, df["ht_score_home"], np.nan)
        df["ht_score_away"] = np.where(is_2nd_half, df["ht_score_away"], np.nan)

    # --- Game state avançado ---
    is_losing = (score_h != score_a).astype(int)
    df["is_home_losing"] = (score_h < score_a).astype(int)
    df["is_away_losing"] = (score_a < score_h).astype(int)

    # Losing team stats
    losing_attacks = np.where(score_h < score_a, att_h,
                      np.where(score_a < score_h, att_a, 0))
    losing_dangerous = np.where(score_h < score_a, da_h,
                        np.where(score_a < score_h, da_a, 0))
    losing_da = losing_dangerous  # alias

    df["losing_team_attack_share"] = np.round(losing_attacks / np.maximum(total_attacks, 1), 4)
    df["losing_team_dangerous_ratio"] = np.round(losing_dangerous / np.maximum(total_dangerous, 1), 4)
    df["urgency_index"] = is_losing * (90 - snap_min_v)
    df["urgency_weighted"] = np.round(is_losing / np.maximum(90 - snap_min_v, 1), 4)

    # Pressure acceleration
    avg_att_rate = total_attacks / np.maximum(snap_min_v, 1)
    df["pressure_acceleration"] = np.round(
        (att_last5 / 5) / np.maximum(avg_att_rate, 0.01), 4)

    # Shares
    df["attack_share_home"] = np.round(att_h / np.maximum(total_attacks, 1), 4)
    df["attack_share_away"] = np.round(att_a / np.maximum(total_attacks, 1), 4)
    df["dangerous_share_home"] = np.round(da_h / np.maximum(total_dangerous, 1), 4)
    df["dangerous_share_away"] = np.round(da_a / np.maximum(total_dangerous, 1), 4)

    # Momentum ratios
    df["corners_momentum_ratio"] = np.round(corners_last_10 / np.maximum(c_total, 1), 4)
    df["attacks_momentum_ratio"] = np.round(att_last10 / np.maximum(total_attacks, 1), 4)
    df["shots_per_dangerous_attack"] = np.round(total_shots / np.maximum(total_dangerous, 1), 4)

    # --- Time since last events ---
    _lcm = df.get("_last_corner_minute", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["time_since_last_corner"] = np.where(
        np.isnan(_lcm), snap_min_v,  # nenhum corner → tempo = snap_min
        snap_min_v - _lcm)
    _lsm = df.get("_last_shot_minute", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["time_since_last_shot"] = np.where(
        np.isnan(_lsm), snap_min_v, snap_min_v - _lsm)
    _ldm = df.get("_last_da_minute", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["time_since_last_dangerous_attack"] = np.where(
        np.isnan(_ldm), snap_min_v, snap_min_v - _ldm)

    # --- Non-linear time ---
    df["snap_minute_sq"] = snap_min_v ** 2
    df["snap_minute_sqrt"] = np.round(np.sqrt(snap_min_v), 4)
    df["snap_minute_log"] = np.round(np.log(np.maximum(snap_min_v, 1)), 4)
    df["remaining_time_sq"] = (90 - snap_min_v) ** 2
    df["time_ratio"] = np.round(snap_min_v / 90, 4)
    df["phase_of_game"] = np.where(snap_min_v <= 30, 0,
                           np.where(snap_min_v <= 60, 1,
                           np.where(snap_min_v <= 75, 2, 3)))
    df["is_last_15min"] = (snap_min_v >= 75).astype(int)
    df["losing_in_last_15"] = ((snap_min_v >= 75) & (score_h != score_a)).astype(int)

    # --- Corner rates by score state ---
    losing_corners = np.where(score_h < score_a, c_home,
                      np.where(score_a < score_h, c_away, 0.0))
    winning_corners = np.where(score_h < score_a, c_away,
                       np.where(score_a < score_h, c_home, 0.0))
    df["losing_team_corners_rate"] = np.round(losing_corners / np.maximum(snap_min_v, 1), 4)
    df["winning_team_corners_rate"] = np.round(winning_corners / np.maximum(snap_min_v, 1), 4)
    df["pressure_when_losing"] = np.round(losing_da / np.maximum(snap_min_v, 1), 4)

    # --- Pressure quality ---
    pressure_ratio = total_dangerous / np.maximum(total_attacks, 1)
    df["pressure_ratio"] = np.round(pressure_ratio, 4)
    df["field_tilt_proxy"] = np.round(da_h / np.maximum(da_h + da_a, 1), 4)

    # Acceleration corners
    df["acceleration_corners"] = np.round(2 * corners_last_5 - corners_last_10, 4)

    # Momentum score
    c_last15 = corners_last_15
    da_last15 = np.nan_to_num(da_l15h, 0) + np.nan_to_num(da_l15a, 0)
    att_last15 = np.nan_to_num(att_l15h, 0) + np.nan_to_num(att_l15a, 0)
    df["momentum_score"] = np.round(3 * c_last15 + 2 * da_last15 + att_last15, 4)

    # --- Game regime ---
    c_rate = df["corners_rate_per_min"].values
    da_rate = df["dangerous_attacks_rate"].values
    sh_rate = total_shots / np.maximum(snap_min_v, 1)
    regime_score = c_rate * 10 + da_rate + sh_rate * 0.5
    game_regime = np.where(regime_score < 0.8, 0, np.where(regime_score < 1.8, 1, 2))
    df["game_regime"] = game_regime

    # --- Regime interactions ---
    df["regime_x_corners_rate"] = np.round(game_regime * c_rate, 4)
    df["regime_x_pressure_home"] = np.round(
        game_regime * df["pressure_index_home"].values, 4)
    df["regime_x_dangerous_rate"] = np.round(game_regime * da_rate, 4)

    exp_at_min_r = np.where(np.isnan(league_avg_v), c_total, league_avg_v * snap_min_v / 90)
    df["is_high_regime_low_corners"] = (
        (game_regime == 2) & (c_total < exp_at_min_r * 0.5)).astype(int)
    df["is_low_regime_high_corners"] = (
        (game_regime == 0) & (c_total > exp_at_min_r * 1.5)).astype(int)

    # --- Dominância ofensiva ---
    df["dangerous_dominance"] = da_h - da_a
    df["corner_dominance"] = c_home - c_away
    df["one_sided_game"] = np.abs(c_home - c_away)

    # --- Expected corners by minute + residual ---
    exp_by_min = np.where(np.isnan(league_avg_v), np.nan,
                          np.round(league_avg_v * snap_min_v / 90, 4))
    df["expected_corners_by_minute"] = exp_by_min
    residual = np.where(np.isnan(exp_by_min), np.nan, np.round(c_total - exp_by_min, 4))
    df["residual_corners"] = residual
    df["overperformance_flag"] = np.where(np.isnan(residual), 0, (residual > 1.5).astype(int))
    df["underperformance_flag"] = np.where(np.isnan(residual), 0, (residual < -1.5).astype(int))

    # --- Intensity drop (2nd half) ---
    fh_rate = first_half_rate
    sh_rate2 = second_half_rate
    df["intensity_drop"] = np.where(is_2nd_half,
        np.round(fh_rate - sh_rate2, 4), np.nan)
    df["second_half_decay_factor"] = np.where(is_2nd_half,
        np.round(sh_rate2 / np.maximum(fh_rate, 0.01), 4), np.nan)
    fatigue = df["fatigue_factor"].values.copy()
    fatigue_safe = np.where(np.isnan(fatigue), 1.0, fatigue)
    df["tempo_adjusted_rate"] = np.where(is_2nd_half,
        np.round(c_rate * fatigue_safe, 4), np.nan)

    # --- Pressure features ---
    df["pressure_diff"] = da_h - da_a
    df["pressure_dominance_ratio"] = np.round(da_h / (da_a + 1), 4)
    df["rolling_pressure_5"] = np.round(da_last5 / 5, 4)
    df["rolling_pressure_10"] = np.round(da_last10 / 10, 4)
    df["da_pressure_acceleration"] = np.round(
        df["rolling_pressure_5"].values - df["rolling_pressure_10"].values, 4)
    df["losing_team_pressure_ratio"] = np.round(losing_da / (total_dangerous + 1), 4)
    df["urgency"] = np.round(np.abs(score_h - score_a) * (snap_min_v / 90), 4)
    df["final_pressure"] = np.round(da_last10 / (90 - snap_min_v + 1), 4)
    df["activity_spike"] = corners_last_5 + shots_last5 + da_last5

    # Corner drought
    tslc = df["time_since_last_corner"].values
    df["corner_drought_pressure"] = np.round(tslc * pressure_ratio, 4)
    df["corner_conversion"] = np.round(c_total / (total_dangerous + 1), 4)
    df["wasted_pressure"] = np.maximum(total_dangerous - total_shots, 0)

    # Pace flags
    league_rate_v = league_avg_v / 90
    df["is_high_pace"] = np.where(np.isnan(league_rate_v), 0,
        (c_rate > league_rate_v).astype(int))
    df["is_low_pace"] = np.where(np.isnan(league_rate_v), 0,
        (c_rate < league_rate_v * 0.7).astype(int))

    # Pace shift
    fh_corners_v = df.get("first_half_corners", pd.Series(np.nan, index=df.index)).values.astype(float)
    fh_rate_shift = np.where(np.isnan(fh_corners_v), 0, fh_corners_v / 45)
    df["pace_shift"] = np.where(
        is_2nd_half & ~np.isnan(fh_corners_v),
        np.round(df["corners_rate_last_10"].values - fh_rate_shift, 4), 0)

    # --- Pressure / momentum / burst ---
    df["pressure_index_5"] = np.round(da_last5 / (att_last5 + 1), 4)
    att_prev5 = att_last10 - att_last5
    da_prev5 = da_last10 - da_last5
    df["acceleration_attacks"] = np.round((att_last5 - att_prev5) / 5, 4)
    df["acceleration_dangerous"] = np.round((da_last5 - da_prev5) / 5, 4)
    df["corners_per_minute_recent"] = np.round(corners_last_5 / 5, 4)

    # Time decay pressure
    tslc_safe = np.maximum(tslc, 0)
    df["time_decay_pressure"] = np.round(
        df["pressure_index_5"].values * np.exp(-tslc_safe / 10), 4)

    # Winning team slowdown
    df["winning_team_slowdown"] = np.where(
        is_home_winning, np.round(att_h / np.maximum(snap_min_v, 1), 4),
        np.where(is_away_winning, np.round(att_a / np.maximum(snap_min_v, 1), 4), 0))

    # Dominance abs
    df["dominance_abs"] = np.abs(
        att_h - att_a + 2 * (da_h - da_a))

    # Efficiency
    df["corners_to_dangerous_ratio"] = np.round(c_total / np.maximum(total_dangerous, 1), 4)
    df["dangerous_to_attacks_ratio"] = np.round(total_dangerous / np.maximum(total_attacks, 1), 4)
    df["shots_to_dangerous_ratio"] = np.round(shots_on / np.maximum(total_dangerous, 1), 4)
    df["conversion_drop"] = np.round(corners_last_5 / (da_last5 + 1), 4)

    # Burst flags
    df["corner_burst_flag"] = (corners_last_5 >= 3).astype(int)
    df["sustained_pressure_flag"] = ((da_last5 >= 8) & (att_last5 >= 15)).astype(int)

    # Expected corners 5min
    lr5 = league_avg_v / 90
    exp_5min = lr5 * 5
    df["expected_corners_5min"] = np.where(np.isnan(lr5), np.nan, np.round(exp_5min, 4))
    df["corners_vs_expected_5"] = np.where(np.isnan(lr5), np.nan,
        np.round(corners_last_5 - exp_5min, 4))

    exp_so_far = league_avg_v * snap_min_v / 90
    df["relative_pace"] = np.where(np.isnan(exp_so_far), np.nan,
        np.round(c_total / np.maximum(exp_so_far, 0.1), 4))

    df["pace_acceleration"] = np.round(
        (corners_last_5 / 5) - (c_total / np.maximum(snap_min_v, 1)), 4)

    # Late game context
    time_frac = snap_min_v / 90
    df["late_game_pressure"] = np.round(df["pressure_index_5"].values * time_frac, 4)
    df["urgency_factor"] = np.round(np.abs(score_h - score_a) * time_frac, 4)
    df["draw_pressure"] = np.round(is_draw * time_frac * da_last5, 4)

    # --- Divergência mercado vs expectativa ---
    cl_v = df.get("corners_line", pd.Series(np.nan, index=df.index)).astype(float).values
    me_v = df["match_expected_corners"].values.astype(float)
    df["line_diff_vs_expected"] = np.where(
        np.isnan(cl_v) | np.isnan(me_v), np.nan,
        np.round(cl_v - me_v, 4))

    # --- Features temporais ---
    kickoff = pd.to_datetime(df.get("kickoff_dt"), errors="coerce")
    df["day_of_week"] = kickoff.dt.dayofweek
    df["hour_of_day"] = kickoff.dt.hour
    df["month"] = kickoff.dt.month
    df["is_weekend"] = (kickoff.dt.dayofweek >= 5).astype(int)

    # --- Targets ---
    ct_final = df.get("corners_total", pd.Series(np.nan, index=df.index)).values
    df["target_corners_total"] = ct_final
    df["target_corners_remaining"] = np.where(
        np.isnan(ct_final), np.nan, ct_final - c_total)
    df["target_more_corners"] = np.where(
        np.isnan(ct_final), np.nan,
        ((ct_final - c_total) > 0).astype(int))
    # Targets separados home/away (para modelos split)
    ch_final = df.get("corners_home_total", pd.Series(np.nan, index=df.index)).values
    ca_final = df.get("corners_away_total", pd.Series(np.nan, index=df.index)).values
    df["target_corners_home_final"] = ch_final
    df["target_corners_away_final"] = ca_final

    # ------------------------------------------------------------------
    # 3. Momentum deltas (entre snapshots consecutivos do mesmo jogo)
    # ------------------------------------------------------------------
    MOMENTUM_COLS = [
        "corners_total_so_far", "corners_home_so_far", "corners_away_so_far",
        "corners_rate_per_min", "dangerous_attacks_home", "dangerous_attacks_away",
        "attacks_home", "attacks_away",
        "shots_on_target_home", "shots_on_target_away",
        "possession_home_avg",
    ]
    MOMENTUM_COLS = [c for c in MOMENTUM_COLS if c in df.columns]
    df = df.sort_values(["event_id", "snap_minute"]).reset_index(drop=True)
    for col in MOMENTUM_COLS:
        df[f"delta_{col}"] = df.groupby("event_id")[col].diff().round(4)

    # ------------------------------------------------------------------
    # 4. Limpeza: remover colunas temporárias
    # ------------------------------------------------------------------
    _tmp_cols = [c for c in df.columns if c.startswith("_") or c.endswith(("_p5", "_p10", "_p15"))]
    df.drop(columns=_tmp_cols, inplace=True, errors="ignore")

    return df


def _last_n_minutes(df, current_min, n, col):
    """Diferença no valor de `col` nos últimos N minutos. (kept for compatibility)"""
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
                                       league_std_map=league_std,
                                       df_h2h=df_h2h)
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
# 4.1 ASSERT DE LEAKAGE TEMPORAL
#
# Valida que features do snapshot do minuto N não carregam dados de minutos >N.
# Qualquer falha aqui é um BLOCKER: o modelo treinado em dados com leakage
# parece ótimo em validação mas colapsa em produção.
# =============================================================================

def assert_no_temporal_leakage(df_features: pd.DataFrame,
                                df_snap: pd.DataFrame,
                                df_pano: pd.DataFrame,
                                snapshot_minutes: list[int],
                                sample_size: int = 50) -> None:
    """
    Verifica 4 invariantes contra vazamento temporal:
      1. corners_total_so_far no snapshot do minuto M == soma de corners no
         df_snap filtrado por minute<=M (verificação direta da fonte)
      2. corners_last_Xmin <= corners_total_so_far (janelas incrementais
         nunca maiores que o acumulado)
      3. Primeiro jogo de cada time tem hist_home_games == 0 e
         hist_home_corners_scored_avg NaN/None (features históricas não
         incluem o próprio jogo)
      4. Primeiro jogo de cada liga tem league_avg_corners None/NaN
    """
    import random
    errors: list[str] = []

    # ---- (1) Reconstruir corners_total_so_far a partir do df_snap bruto ----
    for snap_min in snapshot_minutes:
        df_m = df_features[df_features["snap_minute"] == snap_min]
        if df_m.empty:
            continue
        sample_ids = df_m["event_id"].sample(
            min(sample_size, len(df_m)),
            random_state=42,
        ).tolist()
        for eid in sample_ids:
            row = df_m[df_m["event_id"] == eid].iloc[0]
            saved_csf = row.get("corners_total_so_far")
            if pd.isna(saved_csf):
                continue
            raw = df_snap[(df_snap["event_id"] == eid) & (df_snap["minute"] <= snap_min)]
            if raw.empty:
                continue
            last = raw.sort_values("minute").iloc[-1]
            reconstructed = (float(last.get("corners_home", 0) or 0)
                             + float(last.get("corners_away", 0) or 0))
            if abs(float(saved_csf) - reconstructed) > 0.01:
                errors.append(
                    f"[1] event_id={eid} min={snap_min}: "
                    f"corners_total_so_far={saved_csf} mas df_snap reconstrói {reconstructed}"
                )
                if len(errors) >= 5:
                    break
        if errors:
            break

    # ---- (2) Janelas incrementais <= acumulado ----
    if "corners_total_so_far" in df_features.columns:
        csf = df_features["corners_total_so_far"].fillna(0).values
        for wcol in ["corners_last_5min", "corners_last_10min"]:
            if wcol in df_features.columns:
                w = df_features[wcol].fillna(0).values
                bad = (w > csf + 0.01) & (csf > 0)
                if bad.any():
                    n_bad = int(bad.sum())
                    idx = int(np.argmax(bad))
                    errors.append(
                        f"[2] {wcol}: {n_bad} linhas violam "
                        f"{wcol} <= corners_total_so_far "
                        f"(ex: row {idx}: {w[idx]} > {csf[idx]})"
                    )

    # ---- (3) Primeira aparição global de cada time: hist deve ser 0 ----
    # Importante: build_team_history conta como "histórico do time" TODOS os
    # jogos anteriores (home OU away no passado). Então precisamos encontrar a
    # PRIMEIRA vez (cronologicamente, no panorama) que um team_id aparece em
    # qualquer papel, e validar que o respectivo hist_{home,away}_games é 0.
    if ("hist_home_games" in df_features.columns
            and "hist_away_games" in df_features.columns
            and "home_id" in df_pano.columns
            and "kickoff_dt" in df_pano.columns):
        _pano_sorted = df_pano.copy()
        _pano_sorted["_dt"] = pd.to_datetime(_pano_sorted["kickoff_dt"], errors="coerce")
        _pano_sorted = _pano_sorted.sort_values("_dt").reset_index(drop=True)

        seen: set = set()
        first_appearance: dict[str, tuple[str, str]] = {}  # team_id -> (event_id, role)
        for _, r in _pano_sorted.iterrows():
            h, a = str(r.get("home_id", "")), str(r.get("away_id", ""))
            eid = r.get("event_id")
            if h and h not in seen:
                first_appearance[h] = (eid, "home")
                seen.add(h)
            if a and a not in seen:
                first_appearance[a] = (eid, "away")
                seen.add(a)

        # Usa snapshot mínimo de features para ter uma linha por jogo
        min_snap = min(snapshot_minutes)
        df_min = df_features[df_features["snap_minute"] == min_snap].set_index("event_id")

        violations_3: list[dict] = []
        for team_id, (eid, role) in first_appearance.items():
            if eid not in df_min.index:
                continue
            row = df_min.loc[eid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            col = "hist_home_games" if role == "home" else "hist_away_games"
            val = row.get(col)
            if pd.notna(val) and float(val) != 0:
                violations_3.append({"event_id": eid, "team": team_id,
                                      "role": role, col: float(val)})
                if len(violations_3) >= 5:
                    break

        if violations_3:
            errors.append(
                f"[3] {len(violations_3)}+ primeiras aparições com histórico não-zero. "
                f"Exemplos: {violations_3[:3]}"
            )

    # ---- (4) Primeiro jogo de cada liga: league_avg_corners deve ser NaN ----
    if ("league_avg_corners" in df_features.columns
            and "league_id" in df_pano.columns
            and "kickoff_dt" in df_pano.columns):
        _pano_sorted = df_pano.copy()
        _pano_sorted["_dt"] = pd.to_datetime(_pano_sorted["kickoff_dt"], errors="coerce")
        _pano_sorted = _pano_sorted.sort_values("_dt").reset_index(drop=True)
        first_by_league = (_pano_sorted.dropna(subset=["league_id"])
                           .drop_duplicates(subset=["league_id"], keep="first")
                           [["event_id", "league_id"]])

        min_snap = min(snapshot_minutes)
        df_min = df_features[df_features["snap_minute"] == min_snap].set_index("event_id")

        violations_4: list = []
        for _, r in first_by_league.iterrows():
            eid = r["event_id"]
            if eid not in df_min.index:
                continue
            row = df_min.loc[eid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            val = row.get("league_avg_corners")
            if pd.notna(val):
                violations_4.append({"event_id": eid, "league_id": r["league_id"],
                                      "league_avg_corners": float(val)})
                if len(violations_4) >= 5:
                    break

        if violations_4:
            errors.append(
                f"[4] {len(violations_4)}+ primeiros jogos de ligas com "
                f"league_avg_corners preenchido. Exemplos: {violations_4[:3]}"
            )

    if errors:
        msg = "\n  - ".join(errors)
        raise AssertionError(
            f"LEAKAGE TEMPORAL DETECTADO ({len(errors)} violação(ões)):\n  - {msg}"
        )

    print("  ✓ Nenhuma violação de leakage temporal detectada")


print("\nValidando ausência de leakage temporal...")
assert_no_temporal_leakage(df_features, df_snap, df_pano, SNAPSHOT_MINUTES)

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import xgboost as xgb
    import joblib

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        _OPTUNA = True
    except ImportError:
        _OPTUNA = False
        print("  (optuna não instalado — usando hiperparâmetros fixos. pip install optuna)")

    _SKIP_NGBOOST = "--no-ngboost" in _sys.argv
    _NGBOOST = False
    if not _SKIP_NGBOOST:
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Poisson as NGBPoisson
            from ngboost.scores import CRPScore
            _NGBOOST = True
        except ImportError:
            _NGBOOST = False
            print("  (ngboost não instalado — pip install ngboost)")

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
        "red_cards_home", "red_cards_away",
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
        "expected_remaining_corners", "adjusted_expected_remaining", "remaining_vs_expected",
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
        # Head-to-head (confronto direto)
        "h2h_games", "h2h_corners_avg", "h2h_corners_std",
        "h2h_last3_avg", "h2h_corners_home_avg", "h2h_corners_away_avg",
        "h2h_days_since_last",
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
        # Game regime + interações
        "game_regime",
        "regime_x_corners_rate", "regime_x_pressure_home",
        "regime_x_dangerous_rate",
        "is_high_regime_low_corners", "is_low_regime_high_corners",
        # Força ataque/defesa normalizada pela liga
        "home_attack_strength", "away_attack_strength",
        "home_defense_weakness", "away_defense_weakness",
        # Matchup direto
        "expected_corners_home", "expected_corners_away", "expected_corners_match",
        # Forma recente (últimos 5/10 jogos)
        "home_corners_last5_avg", "away_corners_last5_avg",
        "home_corners_last10_avg", "away_corners_last10_avg",
        # Consistência / volatilidade
        "home_corners_std_last10", "away_corners_std_last10",
        # Ajuste por adversário (schedule strength)
        "home_adjusted_corners", "away_adjusted_corners",
        # Estilo de jogo (proxy)
        "home_corners_per_shot", "away_corners_per_shot",
        "home_corners_per_dangerous", "away_corners_per_dangerous",
        # Desvio em relação à liga
        "home_vs_league", "away_vs_league",
        # Prior pré-live
        "pre_match_expected_total",
        # Sinal precoce (crucial para minuto 15)
        "hist_team_combo_avg", "hist_vs_league", "hist_defensive_strength",
        "no_corner_yet", "early_corner_surge", "first_corner_speed",
        "hist_actual_rate_ratio", "hist_combined_dangerous",
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
        # Pressão / momentum / burst / expectativa dinâmica
        "pressure_index_5",
        "acceleration_attacks", "acceleration_dangerous",
        "corners_per_minute_recent",
        "time_decay_pressure",
        "winning_team_slowdown",
        "dominance_abs",
        "corners_to_dangerous_ratio", "dangerous_to_attacks_ratio",
        "shots_to_dangerous_ratio", "conversion_drop",
        "corner_burst_flag", "sustained_pressure_flag",
        "expected_corners_5min", "corners_vs_expected_5",
        "relative_pace", "pace_acceleration",
        "late_game_pressure", "urgency_factor", "draw_pressure",
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
                df_h2h=df_h2h,
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
        extra_cols: list[str] | None = None,
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

        _extra = [c for c in (extra_cols or []) if c in df.columns and c not in available and c != TARGET]
        df_out = df[available + [TARGET] + _extra].copy()
        df_out = df_out.dropna(subset=[TARGET])

        # Preenche NaN: históricas/liga/h2h/encoding → mediana do treino; ao vivo → 0
        fill_med = [c for c in available if c.startswith(("hist_", "league_", "h2h_"))
                    or c.endswith("_target_enc")]
        fill_zero = [c for c in available if c not in fill_med]

        # Converte colunas object para numérico antes de fillna (evita FutureWarning)
        for c in available:
            if df_out[c].dtype == object:
                df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

        for c in fill_med:
            fill_val = medians[c] if (medians and c in medians) else df_out[c].median()
            df_out[c] = df_out[c].fillna(fill_val)
        for c in fill_zero:
            df_out[c] = df_out[c].fillna(0)

        # Substitui inf/-inf por NaN e depois preenche com 0
        df_out[available] = df_out[available].replace([np.inf, -np.inf], np.nan).fillna(0)

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
        _split_targets = ["target_corners_home_final", "target_corners_away_final"]
        available_train, df_train_clean = prepare_features(
            df_train_raw, feat_cols, extra_cols=_split_targets)

        if len(df_train_clean) < 80:
            print(f"  ⚠ Dados insuficientes no treino ({len(df_train_clean)} amostras). Pulando.")
            continue

        fill_med_cols = [c for c in available_train
                         if c.startswith(("hist_", "league_", "h2h_")) or c.endswith("_target_enc")]
        train_medians = {c: df_train_clean[c].median() for c in fill_med_cols}

        # Aplica as mesmas colunas e medianas do treino ao cal e test
        _, df_cal_clean  = prepare_features(df_cal_raw,  feat_cols,
                                             medians=train_medians,
                                             available_override=available_train,
                                             extra_cols=_split_targets)
        _, df_test_clean = prepare_features(df_test_raw, feat_cols,
                                             medians=train_medians,
                                             available_override=available_train,
                                             extra_cols=_split_targets)

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

        # --- Correlação de cada feature com o target (treino) ---
        _corr_data = X_train.copy()
        _corr_data["_target"] = y_train.values
        _corrs = _corr_data.corr(numeric_only=True)["_target"].drop("_target", errors="ignore")
        _corrs = _corrs.dropna().sort_values(key=abs, ascending=False)
        print(f"\n  Correlação com {TARGET} ({len(_corrs)} features):")
        print(f"    {'Feature':<45s}  {'Corr':>8s}")
        print(f"    {'─'*45}  {'─'*8}")
        for _fname, _cval in _corrs.items():
            print(f"    {_fname:<45s}  {_cval:>+8.4f}")
        print()

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

        # ==================================================================
        # 6c-bis. Modelos home/away separados (A/B vs single-total)
        #
        # Treina dois XGBoost independentes com os mesmos hiperparâmetros
        # e o mesmo feature set, mas targets corners_home_final e
        # corners_away_final. A previsão combinada é a soma.
        # Captura assimetria de pressão entre os times; aceita se MAE
        # combinado for menor ou igual ao single-total.
        # ==================================================================
        model_home_split = None
        model_away_split = None
        mae_split = None
        split_accepted = False
        try:
            if ("target_corners_home_final" in df_train_clean.columns
                    and "target_corners_away_final" in df_train_clean.columns):
                y_train_h = df_train_clean["target_corners_home_final"]
                y_train_a = df_train_clean["target_corners_away_final"]
                y_cal_h   = df_cal_clean["target_corners_home_final"]
                y_cal_a   = df_cal_clean["target_corners_away_final"]
                y_test_h  = df_test_clean["target_corners_home_final"]
                y_test_a  = df_test_clean["target_corners_away_final"]

                _valid_train = y_train_h.notna() & y_train_a.notna()
                _valid_cal   = y_cal_h.notna()   & y_cal_a.notna()
                _valid_test  = y_test_h.notna()  & y_test_a.notna()

                if _valid_train.sum() >= 80 and _valid_test.sum() >= 20:
                    model_home_split = xgb.XGBRegressor(**_best_xgb)
                    model_home_split.fit(
                        X_train[_valid_train.values], y_train_h[_valid_train.values],
                        eval_set=[(X_cal[_valid_cal.values], y_cal_h[_valid_cal.values])],
                        verbose=False,
                    )
                    model_away_split = xgb.XGBRegressor(**_best_xgb)
                    model_away_split.fit(
                        X_train[_valid_train.values], y_train_a[_valid_train.values],
                        eval_set=[(X_cal[_valid_cal.values], y_cal_a[_valid_cal.values])],
                        verbose=False,
                    )

                    pred_split_test = (model_home_split.predict(X_test[_valid_test.values])
                                       + model_away_split.predict(X_test[_valid_test.values]))
                    y_total_test = (y_test_h[_valid_test.values]
                                    + y_test_a[_valid_test.values]).values
                    mae_split = float(mean_absolute_error(y_total_test, pred_split_test))

                    # Comparar com MAE single-total restrito ao mesmo subset
                    mae_single_same = float(mean_absolute_error(
                        y_test[_valid_test.values].values,
                        preds_best[_valid_test.values]))

                    split_accepted = mae_split <= mae_single_same * 1.005  # tolerância 0.5%
                    print(f"\n  Modelo home/away split (A/B vs single-total):")
                    print(f"    MAE single (mesmo subset): {mae_single_same:.4f}")
                    print(f"    MAE split (home+away)    : {mae_split:.4f}")
                    print(f"    → {'ACEITO (split melhor ou igual)' if split_accepted else 'REJEITADO (single melhor)'}")

                    # Salva ambos os modelos
                    joblib.dump(model_home_split,
                                DATA_DIR / f"modelo_corners_xgb_min{snap_min}_home.joblib")
                    joblib.dump(model_away_split,
                                DATA_DIR / f"modelo_corners_xgb_min{snap_min}_away.joblib")
                else:
                    print(f"\n  (split home/away pulado: amostras insuficientes "
                          f"train={_valid_train.sum()} test={_valid_test.sum()})")
        except Exception as _e:
            print(f"\n  (split home/away falhou: {_e})")

        # ==================================================================
        # 6c-ngb. NGBoost com distribuição Poisson (CRPS nativo)
        #
        # Aprende a distribuição diretamente, otimizando CRPS em vez de
        # squared error. A distribuição Poisson é natural para contagens
        # (escanteios). O mu previsto é usado como 6º método probabilístico.
        # ==================================================================
        _ngb_model = None
        _ngb_mu_test = None
        _ngb_mu_cal = None
        _ngb_mae = None
        _ngb_crps = None

        if _NGBOOST:
            try:
                _ngb_model = NGBRegressor(
                    Dist=NGBPoisson,
                    Score=CRPScore,
                    n_estimators=500,
                    learning_rate=0.03,
                    minibatch_frac=0.8,
                    verbose=False,
                    random_state=42,
                    natural_gradient=True,
                )
                _ngb_model.fit(
                    X_train.values, y_train.values.astype(int),
                    X_val=X_cal.values, Y_val=y_cal.values.astype(int),
                    early_stopping_rounds=30,
                )

                # Extrair mu (taxa Poisson prevista)
                _ngb_dist_test = _ngb_model.pred_dist(X_test.values)
                _ngb_dist_cal  = _ngb_model.pred_dist(X_cal.values)

                # Acesso ao parâmetro mu — compatível com ngboost >= 0.5
                try:
                    _ngb_mu_test = np.clip(_ngb_dist_test.params["mu"], 0.01, 60.0)
                    _ngb_mu_cal  = np.clip(_ngb_dist_cal.params["mu"],  0.01, 60.0)
                except (KeyError, TypeError):
                    _ngb_mu_test = np.clip(_ngb_dist_test.mean(), 0.01, 60.0)
                    _ngb_mu_cal  = np.clip(_ngb_dist_cal.mean(),  0.01, 60.0)

                _ngb_mae = float(mean_absolute_error(y_test, _ngb_mu_test))

                # CRPS numérico para Poisson
                _crps_vals = []
                for _yi, _mui in zip(y_test.values, _ngb_mu_test):
                    _k_max = int(max(_mui * 3, _yi * 2, 30))
                    _ks = np.arange(0, _k_max + 1)
                    _cdf = sp_poisson.cdf(_ks, mu=_mui)
                    _ind = (_ks >= _yi).astype(float)
                    _crps_vals.append(float(np.sum((_cdf - _ind) ** 2)))
                _ngb_crps = float(np.mean(_crps_vals))

                print(f"\n  NGBoost (Poisson, CRPScore):")
                print(f"    MAE: {_ngb_mae:.3f}  (vs XGBoost: {mae_best:.3f})")
                print(f"    CRPS (test): {_ngb_crps:.4f}")

            except Exception as _e:
                print(f"\n  (NGBoost treino falhou: {_e})")
                _ngb_model = None
                _ngb_mu_test = None
                _ngb_mu_cal = None

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

        # Sample weights: jogos com resultados extremos recebem mais peso [1, 3]
        _corner_dev = np.abs(y_train.values - dynamic_line_train)
        sample_weights_train = 1.0 + (_corner_dev / max(_corner_dev.max(), 1.0)) * 2.0

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

        # ---- (4) NGBoost Poisson nativo ----
        p_ngb_test = None
        p_ngb_cal  = None
        if _ngb_model is not None and _ngb_mu_test is not None:
            p_ngb_test = np.array([1.0 - sp_poisson.cdf(fl, mu=max(m, 0.01))
                                    for fl, m in zip(fl_test, _ngb_mu_test)])
            p_ngb_cal  = np.array([1.0 - sp_poisson.cdf(fl, mu=max(m, 0.01))
                                    for fl, m in zip(fl_cal, _ngb_mu_cal)])

        # ---- (c) Logística multi-feature ----
        # FIT no TRAIN (não no cal!) para evitar leakage na seleção de método/threshold
        def _logfeat(pred_c, dline, X_df, min_):
            diff = pred_c - dline
            extra = np.column_stack([
                diff,
                diff / np.maximum(dline, 1.0),
                pred_c,
                dline,
            ])
            base = X_df.fillna(0).values.astype(float)
            return np.column_stack([base, extra])

        X_ltrain = _logfeat(preds_train_c, dynamic_line_train, X_train, snap_min)
        X_lcal   = _logfeat(preds_cal_c,   dynamic_line_cal,   X_cal,   snap_min)
        X_ltest  = _logfeat(preds_test_c,  dynamic_line,       X_test,  snap_min)
        if len(np.unique(over_actual_train)) == 2:
            _log_clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
            )
            _log_clf.fit(X_ltrain, over_actual_train,
                         logisticregression__sample_weight=sample_weights_train)
            p_logistic_test = _log_clf.predict_proba(X_ltest)[:, 1]
            p_logistic_cal  = _log_clf.predict_proba(X_lcal)[:, 1]
        else:
            p_logistic_test = p_normal_test.copy()
            p_logistic_cal  = p_normal_cal.copy()

        # ---- (d) XGBoost classificador direto ----
        def _clffeat(pred_c, dline, X_df):
            diff = pred_c - dline
            extra = np.column_stack([diff, pred_c, dline])
            base = X_df.fillna(0).values.astype(float)
            return np.column_stack([base, extra])

        X_clf_tr = _clffeat(preds_train_c, dynamic_line_train, X_train)
        X_clf_ca = _clffeat(preds_cal_c,   dynamic_line_cal,   X_cal)
        X_clf_te = _clffeat(preds_test_c,  dynamic_line,       X_test)

        if len(np.unique(over_actual_train)) == 2:
            _clf_params = dict(
                n_estimators=600, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.4, min_child_weight=3,
                eval_metric="logloss", random_state=42, verbosity=0,
                early_stopping_rounds=30,
            )
            if snap_min <= 20:
                _clf_params.update(
                    n_estimators=1000, max_depth=8, learning_rate=0.01,
                    min_child_weight=1, colsample_bytree=0.6,
                )
            _pos_ratio = over_actual_train.mean()
            if 0.1 < _pos_ratio < 0.9:
                _clf_params["scale_pos_weight"] = round(
                    (1 - _pos_ratio) / _pos_ratio, 4)
            _model_clf = xgb.XGBClassifier(**_clf_params)
            _model_clf.fit(X_clf_tr, over_actual_train,
                           sample_weight=sample_weights_train,
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
        # 6d. Feature importance — SHAP + permutation + seleção iterativa
        #
        # Substitui o feature_importances_ nativo (enviesado pró alta
        # cardinalidade) por:
        #   (1) SHAP TreeExplainer no conjunto de validação
        #   (2) Permutation importance no cal set (sanity check)
        #   (3) Remoção iterativa do bottom 25% e retreino; aceita
        #       se MAE não piorar mais que 1%
        #
        # Artefatos por minuto:
        #   dados_escanteios/shap_summary_min{M}.png
        #   dados_escanteios/feature_ranking_min{M}.csv
        #   dados_escanteios/selected_features_min{M}.json
        # ==================================================================
        _SKIP_SHAP = "--no-shap" in _sys.argv
        selected_features = list(available)  # default: mantém todas

        if not _SKIP_SHAP:
            try:
                import shap as _shap
                from sklearn.inspection import permutation_importance as _perm_imp
                import json as _json

                # --- (1) SHAP no cal set ---
                _explainer = _shap.TreeExplainer(model_mean)
                _shap_vals = _explainer.shap_values(X_cal)
                _shap_abs_mean = np.abs(_shap_vals).mean(axis=0)
                _shap_ser = pd.Series(_shap_abs_mean, index=available).sort_values(ascending=False)

                # --- (2) Permutation importance no cal set ---
                try:
                    _perm = _perm_imp(model_mean, X_cal, y_cal,
                                       n_repeats=5, random_state=42,
                                       scoring="neg_mean_absolute_error")
                    _perm_ser = pd.Series(_perm.importances_mean, index=available)
                except Exception as _e:
                    print(f"  (permutation importance falhou: {_e})")
                    _perm_ser = pd.Series(0.0, index=available)

                # --- Ranking combinado ---
                _ranking = pd.DataFrame({
                    "shap_abs_mean": _shap_ser,
                    "permutation": _perm_ser,
                }).fillna(0).sort_values("shap_abs_mean", ascending=False)
                _ranking.to_csv(DATA_DIR / f"feature_ranking_min{snap_min}.csv")

                print(f"\n  Top 10 features (SHAP):")
                for i, (feat, row_r) in enumerate(_ranking.head(10).iterrows(), 1):
                    print(f"    {i:2d}. {feat:<42s} shap={row_r['shap_abs_mean']:.4f}  "
                          f"perm={row_r['permutation']:+.4f}")

                # --- SHAP summary plot ---
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    _plt.figure()
                    _shap.summary_plot(_shap_vals, X_cal, show=False, max_display=20)
                    _plt.tight_layout()
                    _plt.savefig(DATA_DIR / f"shap_summary_min{snap_min}.png",
                                 dpi=100, bbox_inches="tight")
                    _plt.close("all")
                except Exception as _e:
                    print(f"  (SHAP plot falhou: {_e})")

                # --- (3) Seleção iterativa: remove bottom 25% ---
                n_keep = max(10, int(len(available) * 0.75))
                _candidate_features = _ranking.head(n_keep).index.tolist()
                _removed = [f for f in available if f not in _candidate_features]

                X_train_red = X_train[_candidate_features]
                X_cal_red   = X_cal[_candidate_features]
                X_test_red  = X_test[_candidate_features]

                _model_red = xgb.XGBRegressor(**_best_xgb)
                _model_red.fit(X_train_red, y_train,
                               eval_set=[(X_cal_red, y_cal)], verbose=False)
                _preds_red = _model_red.predict(X_test_red)
                if use_calibration:
                    _cal_red = IsotonicRegression(y_min=0, y_max=35, out_of_bounds="clip")
                    _cal_red.fit(_model_red.predict(X_cal_red), y_cal)
                    _preds_red = _cal_red.predict(_preds_red)
                _mae_red = mean_absolute_error(y_test, _preds_red)

                _tolerance = mae_best * 1.01  # aceita se piorar ≤1%
                _accepted = _mae_red <= _tolerance
                print(f"\n  Feature selection iterativa:")
                print(f"    Features: {len(available)} → {len(_candidate_features)} "
                      f"(removidas {len(_removed)})")
                print(f"    MAE antes: {mae_best:.4f}  MAE depois: {_mae_red:.4f}  "
                      f"({'ACEITO' if _accepted else 'REVERTIDO'})")
                if _accepted:
                    selected_features = _candidate_features

                # --- Salva lista de features selecionadas ---
                with open(DATA_DIR / f"selected_features_min{snap_min}.json", "w") as _fp:
                    _json.dump({
                        "snap_minute": snap_min,
                        "n_original": len(available),
                        "n_selected": len(selected_features),
                        "features": selected_features,
                        "removed": _removed if _accepted else [],
                        "mae_before": round(float(mae_best), 4),
                        "mae_after": round(float(_mae_red), 4),
                        "accepted": bool(_accepted),
                    }, _fp, indent=2)

            except ImportError:
                print(f"\n  (shap não instalado — usando feature_importances_ nativo. "
                      f"pip install shap)")
                feat_imp = pd.Series(model_mean.feature_importances_, index=available)
                top10 = feat_imp.sort_values(ascending=False).head(10)
                print(f"  Top 10 features (gain):")
                for i, (feat, val) in enumerate(top10.items(), 1):
                    print(f"    {i:2d}. {feat:<42s} {val:.4f}")
        else:
            feat_imp = pd.Series(model_mean.feature_importances_, index=available)
            top10 = feat_imp.sort_values(ascending=False).head(10)
            print(f"\n  Top 10 features (gain):")
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
            "selected_features": selected_features,
            "mae_split":         round(mae_split, 4) if mae_split is not None else None,
            "split_accepted":    bool(split_accepted),
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
            "dynamic_line_side":     _bet_side,
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
    print(f"  Estratégia: melhor lado (Over/Under) × método × threshold selecionados no cal set")
    print(f"{'═' * 90}")
    print(f"  {'Min':>4s}  {'Lado':>6s}  {'Linha':>6s}  {'Método':>10s}  {'Thresh':>7s}  "
          f"{'Apostas':>8s}  {'Acurácia':>9s}  {'ROI':>8s}  {'Brier':>7s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}")
    for m, info in all_metadata["models"].items():
        if "dynamic_line_accuracy" not in info:
            continue
        remaining = 90 - m
        linha_formula = f"+{remaining/10:.1f}"
        roi_str    = f"{info['dynamic_line_roi']:+.1%}"
        n_bets     = info.get("dynamic_line_n_bets", "?")
        method_str = info.get("dynamic_line_method", "?")
        thresh_str = f"{info.get('dynamic_line_thresh', 0):.0%}"
        side_str   = info.get("dynamic_line_side", "Over")
        print(f"  {m:>4d}  {side_str:>6s}  {linha_formula:>6s}  {method_str:>10s}  {thresh_str:>7s}  "
              f"{str(n_bets):>8s}  {info['dynamic_line_accuracy']:>8.1%}  "
              f"{roi_str:>8s}  {info['dynamic_line_brier']:>7.4f}")
    print(f"{'═' * 90}")

    # ==================================================================
    # 7. WALK-FORWARD VALIDATION
    #
    # Divide os dados em N folds cronológicos e, para cada fold de teste,
    # treina no passado, calibra no fold anterior e avalia no fold atual.
    # Isso dá estimativas de ROI muito mais confiáveis do que o split único.
    #
    # Folds (N=5):
    #   ti=2: train=fold0,       cal=fold1, test=fold2
    #   ti=3: train=fold0+fold1, cal=fold2, test=fold3
    #   ti=4: train=fold0-2,     cal=fold3, test=fold4
    #
    # Use --no-walkforward para pular esta seção.
    # ==================================================================
    _SKIP_WF = "--no-walkforward" in _sys.argv
    N_WF_FOLDS = 5

    if not _SKIP_WF:
        print(f"\n{'═' * 80}")
        print(f"  WALK-FORWARD VALIDATION  ({N_WF_FOLDS} folds, {N_WF_FOLDS - 2} janelas de teste)")
        print(f"{'═' * 80}")

        # Carrega hparams do cache (já gerado pelo treino acima)
        if _HPARAMS_PATH.exists():
            _wf_hparams = joblib.load(_HPARAMS_PATH)
        else:
            _wf_hparams = {}

        wf_summary: dict[int, dict] = {}

        for snap_min in SNAPSHOT_MINUTES:
            print(f"\n  {'─' * 50}")
            print(f"  ⚽ MINUTO {snap_min}  (walk-forward)")
            print(f"  {'─' * 50}")

            df_min = df_features[df_features["snap_minute"] == snap_min].copy()
            if "kickoff_dt" in df_min.columns:
                df_min = df_min.sort_values("kickoff_dt").reset_index(drop=True)
            n_wf = len(df_min)
            fold_size = n_wf // N_WF_FOLDS

            feat_cols_wf = (BASE_FEATURE_COLS if snap_min == 15
                            else BASE_FEATURE_COLS + MOMENTUM_FEATURE_COLS)
            enc_cols_wf = [c for c in ENCODE_COLS if c in df_min.columns]

            # Hparams do cache (do Optuna do treino principal)
            _hp = _wf_hparams.get(snap_min, {}).get("params")
            if _hp is None:
                _hp = dict(
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0, early_stopping_rounds=30,
                )

            wf_total_bets = 0
            wf_total_wins = 0
            wf_total_profit = 0.0
            wf_fold_rois: list[float] = []
            wf_fold_maes: list[float] = []
            wf_fold_details: list[dict] = []

            for ti in range(2, N_WF_FOLDS):
                # Expanding window: treino cresce a cada fold
                cal_start = (ti - 1) * fold_size
                cal_end = ti * fold_size
                test_start = ti * fold_size
                test_end = (ti + 1) * fold_size if ti < N_WF_FOLDS - 1 else n_wf

                df_tr = df_min.iloc[:cal_start].copy()
                df_ca = df_min.iloc[cal_start:cal_end].copy()
                df_te = df_min.iloc[test_start:test_end].copy()

                if len(df_tr) < 80:
                    print(f"    Fold {ti}: treino insuficiente ({len(df_tr)}). Pulando.")
                    continue

                # Target encoding (fit no treino)
                if enc_cols_wf:
                    _te_wf = TargetEncoderSmoothed(
                        cols=enc_cols_wf, target_col=TARGET, smoothing=10)
                    _te_wf.fit(df_tr)
                    df_tr = _te_wf.transform(df_tr)
                    df_ca = _te_wf.transform(df_ca)
                    df_te = _te_wf.transform(df_te)

                # Feature prep (medianas do treino)
                avail_wf, df_tr_c = prepare_features(df_tr, feat_cols_wf)
                med_wf = {c: df_tr_c[c].median()
                          for c in avail_wf
                          if c.startswith(("hist_", "league_")) or c.endswith("_target_enc")}
                _, df_ca_c = prepare_features(df_ca, feat_cols_wf,
                                              medians=med_wf, available_override=avail_wf)
                _, df_te_c = prepare_features(df_te, feat_cols_wf,
                                              medians=med_wf, available_override=avail_wf)

                if len(df_te_c) < 20 or len(df_ca_c) < 20:
                    print(f"    Fold {ti}: cal/teste insuficiente. Pulando.")
                    continue

                Xtr, ytr = df_tr_c[avail_wf], df_tr_c[TARGET]
                Xca, yca = df_ca_c[avail_wf], df_ca_c[TARGET]
                Xte, yte = df_te_c[avail_wf], df_te_c[TARGET]

                # --- Modelo principal ---
                _m_wf = xgb.XGBRegressor(**_hp)
                _m_wf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)

                _raw_te = _m_wf.predict(Xte)
                _raw_ca = _m_wf.predict(Xca)
                _mae_raw = mean_absolute_error(yte, _raw_te)

                # Calibração isotônica
                _iso_wf = IsotonicRegression(y_min=0, y_max=35, out_of_bounds="clip")
                _iso_wf.fit(_raw_ca, yca)
                _cal_te = _iso_wf.predict(_raw_te)
                _mae_cal = mean_absolute_error(yte, _cal_te)
                _use_cal = _mae_cal < _mae_raw
                _pred_te = _cal_te if _use_cal else _raw_te
                _mae_best = min(_mae_raw, _mae_cal)
                wf_fold_maes.append(_mae_best)

                # Quantile (P10, P90) para sigma
                _q10_wf = xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=0.10,
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.05, reg_lambda=0.5, random_state=42,
                    verbosity=0, early_stopping_rounds=30)
                _q90_wf = xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=0.90,
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.05, reg_lambda=0.5, random_state=42,
                    verbosity=0, early_stopping_rounds=30)
                _q10_wf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
                _q90_wf.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)

                # --- Linha dinâmica ---
                _rem = 90 - snap_min

                def _wf_dline(X_df, rem):
                    csf = (X_df["corners_total_so_far"].values
                           if "corners_total_so_far" in X_df.columns
                           else np.zeros(len(X_df)))
                    la = (X_df["league_avg_corners"].values
                          if "league_avg_corners" in X_df.columns
                          else np.full(len(X_df), np.nan))
                    lines = []
                    for c, l in zip(csf, la):
                        try:
                            r = (float(l) / 90.0
                                 if (l is not None and not np.isnan(float(l))
                                     and float(l) > 0) else 0.1)
                        except (TypeError, ValueError):
                            r = 0.1
                        lines.append(np.round((c + rem * r) * 2) / 2)
                    return np.array(lines, dtype=float)

                dl_te = _wf_dline(Xte, _rem)
                dl_ca = _wf_dline(Xca, _rem)
                dl_tr = _wf_dline(Xtr, _rem)

                oa_te = (yte.values > dl_te).astype(int)
                oa_ca = (yca.values > dl_ca).astype(int)
                oa_tr = (ytr.values > dl_tr).astype(int)

                # Sample weights walk-forward
                _wf_dev = np.abs(ytr.values - dl_tr)
                sw_tr = 1.0 + (_wf_dev / max(_wf_dev.max(), 1.0)) * 2.0

                # Predictions clipped
                _pc_te = np.clip(_pred_te, 0.1, 60.0)
                _cal_ca = _iso_wf.predict(_raw_ca) if _use_cal else _raw_ca
                _pc_ca = np.clip(_cal_ca, 0.1, 60.0)
                _raw_tr = _m_wf.predict(Xtr)
                _pc_tr = np.clip(
                    _iso_wf.predict(_raw_tr) if _use_cal else _raw_tr, 0.1, 60.0)

                fl_te = np.floor(dl_te).astype(int)
                fl_ca = np.floor(dl_ca).astype(int)

                # Sigma heteroscedástico
                _p10_te = _q10_wf.predict(Xte)
                _p90_te = _q90_wf.predict(Xte)
                _p10_ca = _q10_wf.predict(Xca)
                _p90_ca = _q90_wf.predict(Xca)
                sig_te = np.maximum(_p90_te - _p10_te, 1.5) / (2 * 1.28)
                sig_ca = np.maximum(_p90_ca - _p10_ca, 1.5) / (2 * 1.28)

                # --- 5 métodos probabilísticos ---
                # (1) Normal heteroscedástico
                pn_te = 1.0 - sp_norm.cdf(dl_te, loc=_pc_te, scale=sig_te)
                pn_ca = 1.0 - sp_norm.cdf(dl_ca, loc=_pc_ca, scale=sig_ca)

                # (2) Poisson
                pp_te = np.array([1.0 - sp_poisson.cdf(f, mu=max(m, 0.01))
                                  for f, m in zip(fl_te, _pc_te)])
                pp_ca = np.array([1.0 - sp_poisson.cdf(f, mu=max(m, 0.01))
                                  for f, m in zip(fl_ca, _pc_ca)])

                # (3) NegBinom
                _rv = float(np.mean((yca.values - _pc_ca) ** 2))
                _mm = float(np.mean(_pc_ca))
                _nbr = float(np.clip(_mm ** 2 / max(_rv - _mm, 0.5), 0.5, 100.0))
                pnb_te = np.array([
                    1.0 - sp_nbinom.cdf(f, n=_nbr, p=_nbr / (_nbr + max(m, 0.01)))
                    for f, m in zip(fl_te, _pc_te)])
                pnb_ca = np.array([
                    1.0 - sp_nbinom.cdf(f, n=_nbr, p=_nbr / (_nbr + max(m, 0.01)))
                    for f, m in zip(fl_ca, _pc_ca)])

                # (4) Logística multi-feature (fit no TRAIN)
                def _wf_logfeat(pc, dl, X_df, min_):
                    d = pc - dl
                    extra = np.column_stack([
                        d,
                        d / np.maximum(dl, 1.0),
                        pc,
                        dl,
                    ])
                    base = X_df.fillna(0).values.astype(float)
                    return np.column_stack([base, extra])

                if len(np.unique(oa_tr)) == 2:
                    _lc = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
                    )
                    _lc.fit(_wf_logfeat(_pc_tr, dl_tr, Xtr, snap_min), oa_tr,
                            logisticregression__sample_weight=sw_tr)
                    pl_te = _lc.predict_proba(
                        _wf_logfeat(_pc_te, dl_te, Xte, snap_min))[:, 1]
                    pl_ca = _lc.predict_proba(
                        _wf_logfeat(_pc_ca, dl_ca, Xca, snap_min))[:, 1]
                else:
                    pl_te, pl_ca = pn_te.copy(), pn_ca.copy()

                # (5) XGBoost classificador (fit no TRAIN)
                def _wf_clffeat(pc, dl, X_df):
                    diff = pc - dl
                    extra = np.column_stack([diff, pc, dl])
                    base = X_df.fillna(0).values.astype(float)
                    return np.column_stack([base, extra])

                if len(np.unique(oa_tr)) == 2:
                    _xc_params = dict(
                        n_estimators=600, max_depth=6, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.4, min_child_weight=3,
                        eval_metric="logloss", random_state=42, verbosity=0,
                        early_stopping_rounds=30)
                    if snap_min <= 20:
                        _xc_params.update(
                            n_estimators=1000, max_depth=8, learning_rate=0.01,
                            min_child_weight=1, colsample_bytree=0.6)
                    _wf_pos = oa_tr.mean()
                    if 0.1 < _wf_pos < 0.9:
                        _xc_params["scale_pos_weight"] = round(
                            (1 - _wf_pos) / _wf_pos, 4)
                    _xc = xgb.XGBClassifier(**_xc_params)
                    _xc.fit(_wf_clffeat(_pc_tr, dl_tr, Xtr), oa_tr,
                            sample_weight=sw_tr,
                            eval_set=[(_wf_clffeat(_pc_ca, dl_ca, Xca), oa_ca)],
                            verbose=False)
                    pc_te = _xc.predict_proba(
                        _wf_clffeat(_pc_te, dl_te, Xte))[:, 1]
                    pc_ca = _xc.predict_proba(
                        _wf_clffeat(_pc_ca, dl_ca, Xca))[:, 1]
                else:
                    pc_te, pc_ca = pn_te.copy(), pn_ca.copy()

                # Seleção de método (Brier no cal)
                _meths = {
                    "Normal": (pn_te, pn_ca), "Poisson": (pp_te, pp_ca),
                    "NegBinom": (pnb_te, pnb_ca), "Logistic": (pl_te, pl_ca),
                    "XGBClf": (pc_te, pc_ca),
                }
                _briers = {nm: float(np.mean((pcal - oa_ca) ** 2))
                           for nm, (_, pcal) in _meths.items()}
                _bm = min(_briers, key=_briers.get)
                po_te = _meths[_bm][0]
                po_ca = _meths[_bm][1]

                # --- Threshold (Over e Under no cal) ---
                _BE = 1.0 / ODDS_OVER
                _best_ot, _best_or = _BE + MIN_EDGE, -999.0
                _best_ut, _best_ur = _BE + MIN_EDGE, -999.0
                _pu_ca = 1.0 - po_ca
                _ua_ca = 1 - oa_ca

                for _thr in np.arange(_BE + MIN_EDGE, _THRESH_MAX + 0.001, 0.01):
                    # Over
                    _mo = po_ca >= _thr
                    if _mo.sum() >= _MIN_CAL_BET:
                        _wo = oa_ca[_mo].sum()
                        _ro = (_wo * (ODDS_OVER - 1) - (_mo.sum() - _wo)) / _mo.sum()
                        if _ro > _best_or:
                            _best_or, _best_ot = _ro, _thr
                    # Under
                    _mu = _pu_ca >= _thr
                    if _mu.sum() >= _MIN_CAL_BET:
                        _wu = _ua_ca[_mu].sum()
                        _ru = (_wu * (ODDS_UNDER - 1) - (_mu.sum() - _wu)) / _mu.sum()
                        if _ru > _best_ur:
                            _best_ur, _best_ut = _ru, _thr

                if _best_or >= _best_ur and _best_or > 0:
                    _side, _thresh = "Over", _best_ot
                elif _best_ur > 0:
                    _side, _thresh = "Under", _best_ut
                else:
                    _side, _thresh = "Over", _BE + MIN_EDGE

                # --- Aplica ao teste ---
                _pu_te = 1.0 - po_te
                _ua_te = 1 - oa_te
                if _side == "Over":
                    _mask = po_te >= _thresh
                    _act = oa_te
                else:
                    _mask = _pu_te >= _thresh
                    _act = _ua_te

                _nb = int(_mask.sum())
                if _nb > 0:
                    _w = int(_act[_mask].sum())
                    _pf = _w * (ODDS_OVER - 1) - (_nb - _w)
                    _roi = _pf / _nb
                    wf_total_bets += _nb
                    wf_total_wins += _w
                    wf_total_profit += _pf
                    wf_fold_rois.append(_roi)
                    print(f"    Fold {ti}: tr={len(Xtr):,} ca={len(Xca):,} te={len(Xte):,}  "
                          f"MAE={_mae_best:.3f}  {_side} {_bm} {_thresh:.0%}  "
                          f"apostas={_nb}  acur={_w/_nb:.1%}  ROI={_roi:+.1%}")
                else:
                    wf_fold_rois.append(0.0)
                    print(f"    Fold {ti}: tr={len(Xtr):,} ca={len(Xca):,} te={len(Xte):,}  "
                          f"MAE={_mae_best:.3f}  sem apostas qualificadas")

                wf_fold_details.append({
                    "fold": ti, "side": _side, "method": _bm,
                    "thresh": _thresh, "n_bets": _nb,
                    "mae": _mae_best,
                })

            # --- Agregado do minuto ---
            agg_roi = wf_total_profit / wf_total_bets if wf_total_bets > 0 else 0.0
            agg_acc = wf_total_wins / wf_total_bets if wf_total_bets > 0 else 0.0
            agg_mae = float(np.mean(wf_fold_maes)) if wf_fold_maes else 0.0
            _nonzero_rois = [r for r in wf_fold_rois if r != 0.0]
            roi_std = float(np.std(_nonzero_rois)) if len(_nonzero_rois) >= 2 else 0.0

            wf_summary[snap_min] = {
                "total_bets": wf_total_bets,
                "accuracy": round(agg_acc, 4),
                "roi": round(agg_roi, 4),
                "roi_std": round(roi_std, 4),
                "fold_rois": [round(r, 4) for r in wf_fold_rois],
                "mae_avg": round(agg_mae, 4),
                "n_folds": len(wf_fold_maes),
            }

            print(f"\n    Agregado min {snap_min}: apostas={wf_total_bets:,}  "
                  f"acur={agg_acc:.1%}  ROI={agg_roi:+.1%}  "
                  f"MAE={agg_mae:.3f}  ROI std={roi_std:.1%}")

        # --- Tabela resumo Walk-Forward ---
        print(f"\n{'═' * 90}")
        print(f"  WALK-FORWARD VALIDATION — RESUMO AGREGADO ({N_WF_FOLDS} folds, "
              f"{N_WF_FOLDS - 2} janelas)")
        print(f"{'═' * 90}")
        print(f"  {'Min':>4s}  {'Apostas':>8s}  {'Acurácia':>9s}  {'ROI':>8s}  "
              f"{'ROI std':>8s}  {'MAE':>6s}  {'Folds':>5s}  {'ROI por fold'}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*30}")
        for m in SNAPSHOT_MINUTES:
            if m not in wf_summary:
                continue
            info = wf_summary[m]
            fr_str = ", ".join(f"{r:+.1%}" for r in info["fold_rois"])
            print(f"  {m:>4d}  {info['total_bets']:>8,}  {info['accuracy']:>8.1%}  "
                  f"{info['roi']:>+7.1%}  {info['roi_std']:>7.1%}  "
                  f"{info['mae_avg']:>6.3f}  {info['n_folds']:>5d}  {fr_str}")
        print(f"{'═' * 90}")
        print(f"\n  Interpretação:")
        print(f"    - ROI consistente entre folds (std < 10%) → sinal confiável")
        print(f"    - ROI com alta variância entre folds → possível overfitting no threshold")
        print(f"    - ROI walk-forward < ROI single-split → o split único estava otimista")
        print(f"  (use --no-walkforward para pular esta seção)")

        # Salva no metadata
        all_metadata["walkforward"] = wf_summary
        joblib.dump(all_metadata, DATA_DIR / "modelo_corners_meta.joblib")

except ImportError as e:
    print(f"  Pacote não instalado ({e}). Instale: pip install xgboost scikit-learn joblib")

# %%
print("\nAnálise concluída!")
