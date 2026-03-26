"""
=============================================================================
BetsAPI – Previsão de Escanteios em Tempo Real
=============================================================================
Carrega o modelo XGBoost treinado e faz previsões para jogos ao vivo,
exibindo uma tabela formatada no terminal a cada ciclo.

Uso:
    python betsapi_corners_predictor.py --token SEU_TOKEN
    python betsapi_corners_predictor.py --token SEU_TOKEN --interval 60 --min-edge 0.5
=============================================================================
"""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from betsapi_corners_collector import (
    BetsAPIClient,
    extract_event_metadata,
    parse_corner_odds,
    parse_stats_trend,
    REQUEST_DELAY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("dados_escanteios")
MODEL_PATH = DATA_DIR / "modelo_corners_xgb.joblib"
PANORAMA_PATH = DATA_DIR / "panorama_jogos.parquet"
PREDICOES_PATH = DATA_DIR / "predicoes_live.csv"
RELOAD_EVERY_N_CYCLES = 15  # recarrega panorama a cada N ciclos

# Deve espelhar exatamente o FEATURE_COLS do betsapi_corners_analysis.py
FEATURE_COLS: list[str] = [
    "snap_minute",
    "corners_home_so_far", "corners_away_so_far", "corners_total_so_far",
    "corners_rate_per_min", "corners_diff",
    "corners_last_15_home", "corners_last_15_away",
    "corners_per_attack_ratio",
    "possession_home_avg", "possession_away_avg",
    "attacks_home", "attacks_away",
    "dangerous_attacks_home", "dangerous_attacks_away",
    "attacks_diff", "dangerous_attacks_diff",
    "dangerous_attacks_rate",
    "shots_on_target_home", "shots_on_target_away",
    "shots_off_target_home", "shots_off_target_away",
    "shots_on_target_diff",
    "yellow_cards_home", "yellow_cards_away",
    "red_cards_home", "red_cards_away",
    "fouls_home", "fouls_away",
    "saves_home", "saves_away",
    "offsides_home", "offsides_away",
    "goal_kicks_home", "goal_kicks_away",
    "score_home", "score_away",
    "score_diff", "total_red_cards", "red_card_diff",
    "corners_line", "corners_over_odds", "corners_under_odds",
    "asian_corners_line",
    "odds_home_win", "odds_draw", "odds_away_win",
    "goals_line", "goals_over_odds", "goals_under_odds",
    "btts_yes_odds", "btts_no_odds",
    "h2h_total_games", "h2h_avg_corners_total",
    "h2h_avg_corners_home", "h2h_avg_corners_away",
    "h2h_avg_goals_total",
    "hist_home_corners_avg", "hist_away_corners_avg",
    "hist_home_corners_scored_avg", "hist_away_corners_scored_avg",
    "hist_home_corners_conceded_avg", "hist_away_corners_conceded_avg",
    "hist_home_corners_home_avg", "hist_away_corners_away_avg",
    "hist_home_dangerous_attacks_avg", "hist_away_dangerous_attacks_avg",
    "hist_home_goals_avg", "hist_away_goals_avg",
    "hist_home_games", "hist_away_games",
    "days_rest_home", "days_rest_away",
    "league_avg_corners",
    "day_of_week", "hour_of_day", "month", "is_weekend",
]


# ---------------------------------------------------------------------------
# Cache de histórico dos times e médias de liga
# ---------------------------------------------------------------------------

class HistoryCache:
    """
    Precomputa rolling averages de times e médias de liga a partir do panorama.
    Alimenta as features históricas para previsões ao vivo.
    """

    _STAT_COLS: dict[str, str] = {
        "corners_total":                "corners",
        "corners_home_total":           "corners_scored",
        "corners_away_total":           "corners_conceded",
        "dangerous_attacks_home_total": "dangerous_attacks",
        "attacks_home_total":           "attacks",
        "shots_on_target_home_total":   "shots_on_target",
        "total_goals":                  "goals",
    }

    def __init__(self, df_pano: pd.DataFrame, window: int = 10):
        self._window = window
        self._team_history: dict[str, list[dict]] = {}
        self._last_game_date: dict[str, pd.Timestamp] = {}
        self._league_history: dict[str, list[float]] = {}
        self._build(df_pano)

    def _build(self, df_pano: pd.DataFrame) -> None:
        if df_pano.empty:
            return
        df = df_pano.copy()
        df["_dt"] = pd.to_datetime(df.get("kickoff_dt", pd.Series(dtype=str)), errors="coerce")
        df = df.sort_values("_dt").reset_index(drop=True)

        for _, row in df.iterrows():
            home_id = str(row.get("home_id", ""))
            away_id = str(row.get("away_id", ""))
            league_id = str(row.get("league_id", ""))
            dt = row.get("_dt")

            home_stats: dict = {col: row.get(col) for col in self._STAT_COLS if col in df.columns}
            home_stats["_is_home"] = True

            away_stats = dict(home_stats)
            away_stats["corners_home_total"] = row.get("corners_away_total")
            away_stats["corners_away_total"] = row.get("corners_home_total")
            away_stats["_is_home"] = False

            if home_id:
                self._team_history.setdefault(home_id, []).append(home_stats)
                if pd.notna(dt):
                    self._last_game_date[home_id] = dt
            if away_id:
                self._team_history.setdefault(away_id, []).append(away_stats)
                if pd.notna(dt):
                    self._last_game_date[away_id] = dt

            ct = row.get("corners_total")
            if league_id and ct is not None:
                self._league_history.setdefault(league_id, []).append(float(ct))

    def _team_features(self, team_id: str, role: str, current_dt=None) -> dict:
        """role = 'home' or 'away'."""
        feat: dict = {}
        is_home_role = (role == "home")

        if team_id in self._team_history:
            hist = self._team_history[team_id][-self._window:]
            for col, alias in self._STAT_COLS.items():
                vals = [h.get(col) for h in hist if h.get(col) is not None]
                feat[f"hist_{role}_{alias}_avg"] = round(float(np.mean(vals)), 2) if vals else None
            feat[f"hist_{role}_games"] = len(hist)

            # Mando de campo
            mando = [h for h in hist if h.get("_is_home") is is_home_role]
            mc_vals = [h.get("corners_home_total") for h in mando
                       if h.get("corners_home_total") is not None]
            key = f"hist_{role}_corners_{'home' if is_home_role else 'away'}_avg"
            feat[key] = round(float(np.mean(mc_vals)), 2) if mc_vals else None

            # Dias de descanso
            if current_dt is not None and team_id in self._last_game_date:
                last_dt = self._last_game_date[team_id]
                if pd.notna(last_dt) and pd.notna(current_dt):
                    feat[f"days_rest_{role}"] = int((current_dt - last_dt).days)
                else:
                    feat[f"days_rest_{role}"] = None
            else:
                feat[f"days_rest_{role}"] = None
        else:
            for alias in self._STAT_COLS.values():
                feat[f"hist_{role}_{alias}_avg"] = None
            feat[f"hist_{role}_games"] = 0
            feat[f"hist_{role}_corners_{'home' if is_home_role else 'away'}_avg"] = None
            feat[f"days_rest_{role}"] = None

        return feat

    def get_home_features(self, home_id: str, current_dt=None) -> dict:
        return self._team_features(home_id, "home", current_dt)

    def get_away_features(self, away_id: str, current_dt=None) -> dict:
        return self._team_features(away_id, "away", current_dt)

    def get_league_avg(self, league_id: str) -> Optional[float]:
        hist = self._league_history.get(str(league_id), [])
        return round(sum(hist) / len(hist), 2) if hist else None


# ---------------------------------------------------------------------------
# Construção de features para um snapshot ao vivo
# ---------------------------------------------------------------------------

def _mean_col(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return round(float(vals.mean()), 2) if len(vals) > 0 else None


def _last_n_minutes(snap_df: pd.DataFrame, current_min: int, n: int,
                    col: str) -> Optional[int]:
    if col not in snap_df.columns:
        return None
    since = current_min - n
    past = snap_df[snap_df["minute"] <= since]
    recent = snap_df[snap_df["minute"] <= current_min]
    if past.empty or recent.empty:
        return None
    vp = past.iloc[-1].get(col)
    vn = recent.iloc[-1].get(col)
    if vp is None or vn is None:
        return None
    return int(vn) - int(vp)


def _diff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def build_live_snapshot_features(
    snapshot_rows: list[dict],
    meta: dict,
    snap_minute: int,
    hist_cache: HistoryCache,
    odds_dict: Optional[dict] = None,
) -> dict:
    """
    Constrói um dict de features para um jogo ao vivo no minuto `snap_minute`.
    Retorna {} se não há snapshots suficientes.
    """
    if not snapshot_rows:
        return {}

    snap_df = pd.DataFrame(snapshot_rows).sort_values("minute").reset_index(drop=True)
    until = snap_df[snap_df["minute"] <= snap_minute]
    if until.empty:
        until = snap_df

    last = until.iloc[-1]
    actual_min = int(last.get("minute") or snap_minute)

    c_home = last.get("corners_home") or 0
    c_away = last.get("corners_away") or 0
    att_home = last.get("attacks_home") or 0
    att_away = last.get("attacks_away") or 0
    da_home = last.get("dangerous_attacks_home") or 0
    da_away = last.get("dangerous_attacks_away") or 0
    rc_home = last.get("red_cards_home") or 0
    rc_away = last.get("red_cards_away") or 0

    kickoff = pd.to_datetime(meta.get("kickoff_dt"), errors="coerce")
    kickoff_ts = kickoff if pd.notna(kickoff) else None

    feat: dict = {
        "snap_minute": actual_min,
        # Escanteios
        "corners_home_so_far":      c_home,
        "corners_away_so_far":      c_away,
        "corners_total_so_far":     c_home + c_away,
        "corners_rate_per_min":     round((c_home + c_away) / max(actual_min, 1), 4),
        "corners_diff":             _diff(last.get("corners_home"), last.get("corners_away")),
        "corners_last_15_home":     _last_n_minutes(snap_df, actual_min, 15, "corners_home"),
        "corners_last_15_away":     _last_n_minutes(snap_df, actual_min, 15, "corners_away"),
        "corners_per_attack_ratio": round((c_home + c_away) / max(att_home + att_away, 1), 4),
        # Posse
        "possession_home_avg":      _mean_col(until, "possession_home"),
        "possession_away_avg":      _mean_col(until, "possession_away"),
        # Ataques
        "attacks_home":             last.get("attacks_home"),
        "attacks_away":             last.get("attacks_away"),
        "dangerous_attacks_home":   last.get("dangerous_attacks_home"),
        "dangerous_attacks_away":   last.get("dangerous_attacks_away"),
        "attacks_diff":             _diff(last.get("attacks_home"), last.get("attacks_away")),
        "dangerous_attacks_diff":   _diff(da_home, da_away),
        "dangerous_attacks_rate":   round((da_home + da_away) / max(actual_min, 1), 4),
        # Chutes
        "shots_on_target_home":     last.get("shots_on_target_home"),
        "shots_on_target_away":     last.get("shots_on_target_away"),
        "shots_off_target_home":    last.get("shots_off_target_home"),
        "shots_off_target_away":    last.get("shots_off_target_away"),
        "shots_on_target_diff":     _diff(last.get("shots_on_target_home"),
                                          last.get("shots_on_target_away")),
        # Cartões e faltas
        "yellow_cards_home":        last.get("yellow_cards_home"),
        "yellow_cards_away":        last.get("yellow_cards_away"),
        "red_cards_home":           rc_home,
        "red_cards_away":           rc_away,
        "fouls_home":               last.get("fouls_home"),
        "fouls_away":               last.get("fouls_away"),
        # Saves / offsides / goal kicks
        "saves_home":               last.get("saves_home"),
        "saves_away":               last.get("saves_away"),
        "offsides_home":            last.get("offsides_home"),
        "offsides_away":            last.get("offsides_away"),
        "goal_kicks_home":          last.get("goal_kicks_home"),
        "goal_kicks_away":          last.get("goal_kicks_away"),
        # Placar e contexto
        "score_home":               last.get("score_home"),
        "score_away":               last.get("score_away"),
        "score_diff":               _diff(last.get("score_home"), last.get("score_away")),
        "total_red_cards":          rc_home + rc_away,
        "red_card_diff":            _diff(rc_home, rc_away),
        # Odds pré-jogo (se disponíveis)
        "corners_line":             (odds_dict or {}).get("corners_line"),
        "corners_over_odds":        (odds_dict or {}).get("corners_over_odds"),
        "corners_under_odds":       (odds_dict or {}).get("corners_under_odds"),
        "asian_corners_line":       (odds_dict or {}).get("asian_corners_line"),
        "odds_home_win":            None,
        "odds_draw":                None,
        "odds_away_win":            None,
        "goals_line":               None,
        "goals_over_odds":          None,
        "goals_under_odds":         None,
        "btts_yes_odds":            None,
        "btts_no_odds":             None,
        # H2H — não disponível ao vivo
        "h2h_total_games":          None,
        "h2h_avg_corners_total":    None,
        "h2h_avg_corners_home":     None,
        "h2h_avg_corners_away":     None,
        "h2h_avg_goals_total":      None,
        # Liga
        "league_avg_corners":       hist_cache.get_league_avg(str(meta.get("league_id", ""))),
        # Temporal
        "day_of_week":  kickoff_ts.dayofweek if kickoff_ts is not None else None,
        "hour_of_day":  kickoff_ts.hour      if kickoff_ts is not None else None,
        "month":        kickoff_ts.month     if kickoff_ts is not None else None,
        "is_weekend":   int(kickoff_ts.dayofweek >= 5) if kickoff_ts is not None else None,
    }

    feat.update(hist_cache.get_home_features(str(meta.get("home_id", "")), kickoff_ts))
    feat.update(hist_cache.get_away_features(str(meta.get("away_id", "")), kickoff_ts))

    return feat


# ---------------------------------------------------------------------------
# Execução de um ciclo de previsão
# ---------------------------------------------------------------------------

def run_predictor_cycle(
    client: BetsAPIClient,
    model,
    hist_cache: HistoryCache,
    min_edge: float = 0.0,
) -> list[dict]:
    """
    Busca jogos ao vivo, constrói features e retorna lista de dicts de previsão.
    Cada dict contém os dados para exibição e logging.
    """
    inplay_resp = client.get_inplay_events()
    time.sleep(REQUEST_DELAY)
    live_events = inplay_resp.get("results", []) or []

    predictions: list[dict] = []

    for event in live_events:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue

        meta = extract_event_metadata(event)
        snap_minute = int(meta.get("match_minute") or 0)
        if snap_minute <= 0:
            continue  # pré-jogo ou sem minuto

        # Stats trend
        trend_resp = client.get_stats_trend(event_id)
        time.sleep(REQUEST_DELAY)
        raw_trend = trend_resp.get("results", []) or []
        snapshot_rows = parse_stats_trend(raw_trend, event_id, meta)
        if not snapshot_rows:
            continue

        # Odds pré-jogo (best-effort)
        odds_resp = client.get_prematch_odds(event_id)
        time.sleep(REQUEST_DELAY)
        odds_dict = parse_corner_odds(odds_resp)

        # Constrói features
        feat = build_live_snapshot_features(
            snapshot_rows, meta, snap_minute, hist_cache, odds_dict
        )
        if not feat:
            continue

        # Monta vetor de entrada (NaN para features ausentes)
        row_vals = [feat.get(c) for c in FEATURE_COLS]
        X = pd.DataFrame([row_vals], columns=FEATURE_COLS).astype(float)

        try:
            pred_total = float(model.predict(X)[0])
        except Exception as exc:
            log.warning("Erro ao predizer evento %s: %s", event_id, exc)
            continue

        c_home = feat.get("corners_home_so_far") or 0
        c_away = feat.get("corners_away_so_far") or 0
        corners_now = c_home + c_away
        remaining = max(round(pred_total - corners_now, 1), 0)

        corners_line = feat.get("corners_line")
        edge = round(pred_total - corners_line, 2) if corners_line is not None else None

        if min_edge > 0 and (edge is None or abs(edge) < min_edge):
            continue

        predictions.append({
            "timestamp":   datetime.utcnow().isoformat(timespec="seconds"),
            "event_id":    event_id,
            "league_name": meta.get("league_name", ""),
            "league_id":   meta.get("league_id", ""),
            "home_team":   meta.get("home_team", ""),
            "away_team":   meta.get("away_team", ""),
            "minute":      snap_minute,
            "score_home":  meta.get("score_home"),
            "score_away":  meta.get("score_away"),
            "corners_home": c_home,
            "corners_away": c_away,
            "corners_now":  corners_now,
            "pred_total":   round(pred_total, 1),
            "remaining":    remaining,
            "corners_line": corners_line,
            "over_odds":    feat.get("corners_over_odds"),
            "under_odds":   feat.get("corners_under_odds"),
            "edge":         edge,
            "bet":          ("OVER" if (edge or 0) > 0 else "UNDER") if edge is not None else None,
        })

    # Ordena por maior |edge|
    predictions.sort(key=lambda x: abs(x.get("edge") or 0), reverse=True)
    return predictions


# ---------------------------------------------------------------------------
# Exibição e logging
# ---------------------------------------------------------------------------

def format_table(predictions: list[dict], cycle: int) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 100
    header = f" BetsAPI Corners Predictor  |  Ciclo {cycle}  |  {now} UTC"
    col_hdr = (
        f"  {'Partida':<30} {'Min':>4}  {'Sc':>5}  {'Cant':>5}  "
        f"{'Prev':>5}  {'Rest':>5}  {'Linha':>6}  {'Edge':>8}  Direção"
    )
    divider = "-" * 100
    lines = [sep, header, divider, col_hdr, divider]

    for p in predictions:
        score = f"{p['score_home']}-{p['score_away']}" if p.get("score_home") is not None else "?-?"
        match_name = f"{p['home_team']} vs {p['away_team']}"[:30]
        linha_str = f"{p['corners_line']:.1f}" if p.get("corners_line") is not None else "  N/A"
        edge_str  = f"{p['edge']:+.2f}"        if p.get("edge")         is not None else "   N/A"
        bet_str   = p.get("bet") or ""
        flag = "◄ VALUE" if p.get("edge") is not None and abs(p["edge"]) >= 1.5 else ""
        lines.append(
            f"  {match_name:<30} {p['minute']:>4}  {score:>5}  "
            f"{p['corners_now']:>5}  {p['pred_total']:>5.1f}  "
            f"{p['remaining']:>5.1f}  {linha_str:>6}  {edge_str:>8}  "
            f"{bet_str:<6} {flag}"
        )

    lines.append(divider)
    lines.append(
        f"  {len(predictions)} jogo(s) exibido(s) | "
        f"{sum(1 for p in predictions if p.get('edge') is not None)} com linha de odds"
    )
    lines.append(sep)
    return "\n".join(lines)


def save_predictions(predictions: list[dict], path: Path) -> None:
    if not predictions:
        return
    fieldnames = list(predictions[0].keys())
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(predictions)


# ---------------------------------------------------------------------------
# Helpers de inicialização (importáveis por betsapi_corners_alerts.py)
# ---------------------------------------------------------------------------

def load_model(path: Path = MODEL_PATH):
    """Carrega o modelo joblib. Lança FileNotFoundError se não existir."""
    import joblib
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute primeiro: python betsapi_corners_analysis.py"
        )
    model = joblib.load(path)
    log.info("Modelo carregado: %s", path)
    return model


def load_history_cache(path: Path = PANORAMA_PATH, window: int = 10) -> HistoryCache:
    """Carrega panorama e constrói o HistoryCache."""
    if not path.exists():
        log.warning("Panorama não encontrado (%s) — histórico vazio.", path)
        return HistoryCache(pd.DataFrame(), window)
    df_pano = pd.read_parquet(path)
    cache = HistoryCache(df_pano, window)
    log.info("HistoryCache construído: %d times, %d ligas",
             len(cache._team_history), len(cache._league_history))
    return cache


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Previsão ao vivo de escanteios (BetsAPI)")
    p.add_argument("--token",    required=True, help="Token da BetsAPI")
    p.add_argument("--interval", type=int,   default=60,  help="Intervalo entre ciclos (s)")
    p.add_argument("--min-edge", type=float, default=0.0, help="Edge mínimo para exibir")
    p.add_argument("--window",   type=int,   default=10,  help="Rolling window do histórico")
    p.add_argument("--output",   default=str(PREDICOES_PATH), help="CSV de saída")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = BetsAPIClient(token=args.token)
    model  = load_model()
    cache  = load_history_cache(window=args.window)
    out    = Path(args.output)

    cycle = 0
    log.info("Iniciando previsão ao vivo (intervalo=%ds, min-edge=%.1f). Ctrl+C para parar.",
             args.interval, args.min_edge)

    try:
        while True:
            cycle += 1

            # Recarrega cache periodicamente (novos jogos finalizados)
            if cycle % RELOAD_EVERY_N_CYCLES == 0:
                log.info("Recarregando HistoryCache...")
                cache = load_history_cache(window=args.window)

            predictions = run_predictor_cycle(client, model, cache, min_edge=args.min_edge)
            print(format_table(predictions, cycle))

            save_predictions(predictions, out)
            if predictions:
                log.info("Ciclo %d: %d previsões salvas em %s", cycle, len(predictions), out)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("Encerrado pelo usuário.")


if __name__ == "__main__":
    main()
