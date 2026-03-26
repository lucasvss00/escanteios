"""
=============================================================================
BetsAPI – Coletor de Dados para Modelo de Escanteios
=============================================================================
Modos de operação:
  1. HISTÓRICO  → coleta jogos finalizados (por dia ou range de datas)
  2. AO VIVO    → monitora jogos em andamento em loop contínuo

Dados coletados por minuto (stats_trend):
  - Escanteios acumulados (home / away)
  - Posse de bola (home / away)
  - Chutes a gol / fora (on_target / off_target)
  - Cartões amarelos / vermelhos
  - Faltas
  - Ataques
  - Ataques perigosos

Saída:
  - {output_dir}/snapshots_por_minuto.parquet   ← série temporal por minuto
  - {output_dir}/panorama_jogos.parquet          ← 1 linha por jogo finalizado
  - {output_dir}/snapshots_por_minuto.csv        ← cópia CSV opcional
  - {output_dir}/panorama_jogos.csv              ← cópia CSV opcional

Uso rápido:
  pip install requests pandas pyarrow tqdm

  # Histórico – últimos 3 dias:
  python betsapi_corners_collector.py --mode historico --days 3 --token SEU_TOKEN

  # Ao vivo – polling a cada 60s:
  python betsapi_corners_collector.py --mode live --interval 60 --token SEU_TOKEN

  # Histórico – range específico:
  python betsapi_corners_collector.py --mode historico \
      --start 20240101 --end 20240131 --token SEU_TOKEN
=============================================================================
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_URL = "https://api.b365api.com"
SPORT_ID = 1          # Futebol
MAX_RETRIES = 3
RETRY_DELAY = 5       # segundos entre tentativas
REQUEST_DELAY = 0.35  # delay entre requests — 3600 req/hora ≈ 3 req/s (conservador)


# ---------------------------------------------------------------------------
# Exceção de rate limit
# ---------------------------------------------------------------------------

class RateLimitReached(Exception):
    """Disparada quando o contador de requests atinge o limite configurado."""
    pass


# ---------------------------------------------------------------------------
# Cliente HTTP
# ---------------------------------------------------------------------------
class BetsAPIClient:
    """Wrapper simples para a BetsAPI com retry automático e controle de rate limit."""

    def __init__(self, token: str, max_requests: int = 3500, auto_wait: bool = True):
        self.token        = token
        self.session      = requests.Session()
        self.session.headers.update({"X-API-TOKEN": token})
        self.max_requests = max_requests   # limite por janela de 1h
        self.auto_wait    = auto_wait      # pausa automática ao atingir o limite
        self.request_count = 0
        self.window_start  = datetime.utcnow()

    @property
    def requests_remaining(self) -> int:
        return max(0, self.max_requests - self.request_count)

    def seconds_until_reset(self) -> int:
        elapsed = (datetime.utcnow() - self.window_start).total_seconds()
        return max(0, int(3600 - elapsed))

    def _reset_window_if_needed(self):
        elapsed = (datetime.utcnow() - self.window_start).total_seconds()
        if elapsed >= 3600:
            log.info("Nova janela de rate limit iniciada (anterior: %d requests em %.0fs).",
                     self.request_count, elapsed)
            self.request_count = 0
            self.window_start  = datetime.utcnow()

    def _check_rate_limit(self):
        self._reset_window_if_needed()
        if self.request_count < self.max_requests:
            return  # dentro do limite, ok

        if self.auto_wait:
            # Calcula tempo restante + 15s de margem
            wait_sec = self.seconds_until_reset() + 15
            print(f"\n{'='*60}")
            print(f"  ⏸  RATE LIMIT ATINGIDO ({self.request_count}/{self.max_requests} requests)")
            print(f"  Aguardando {wait_sec // 60} min {wait_sec % 60} s para janela resetar...")
            print(f"  Pressione Ctrl+C para parar e salvar checkpoint.")
            print(f"{'='*60}")
            for i in range(1, wait_sec + 1):
                time.sleep(1)
                if i % 60 == 0:   # atualiza a cada 1 minuto
                    remaining = wait_sec - i
                    print(f"  ⏳ Aguardando... {remaining // 60} min {remaining % 60} s restantes")
            # Reseta janela e continua normalmente
            self.request_count = 0
            self.window_start  = datetime.utcnow()
            print(f"\n  ▶  Janela resetada — retomando coleta...\n")
        else:
            elapsed  = (datetime.utcnow() - self.window_start).total_seconds()
            wait_sec = self.seconds_until_reset()
            raise RateLimitReached(
                f"Limite de {self.max_requests} requests atingido "
                f"(janela: {elapsed:.0f}s). "
                f"Próxima janela em ~{wait_sec}s. Use --resume para retomar."
            )

    def _get(self, endpoint: str, params: dict = None) -> dict:
        self._check_rate_limit()   # lança RateLimitReached se necessário

        url = f"{BASE_URL}{endpoint}"
        params = params or {}
        params.setdefault("token", self.token)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                self.request_count += 1
                if data.get("success") == 1:
                    return data
                log.warning("API retornou success=0 em %s: %s", endpoint, data.get("error", ""))
                return data
            except requests.RequestException as exc:
                log.warning("Tentativa %d/%d falhou (%s): %s", attempt, MAX_RETRIES, endpoint, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        return {}

    # ------------------------------------------------------------------
    # Endpoints de eventos
    # ------------------------------------------------------------------

    def get_ended_events(self, day: str, page: int = 1, league_id: Optional[int] = None) -> dict:
        """
        Jogos finalizados de um dia específico.
        day: formato YYYYMMDD
        """
        params = {"sport_id": SPORT_ID, "day": day, "page": page}
        if league_id:
            params["league_id"] = league_id
        return self._get("/v3/events/ended", params)

    def get_inplay_events(self) -> dict:
        """Jogos ao vivo no momento."""
        return self._get("/v3/events/inplay", {"sport_id": SPORT_ID})

    def get_event_view(self, event_id: str) -> dict:
        """Detalhes e placar final de um evento."""
        return self._get("/v1/event/view", {"event_id": event_id})

    def get_stats_trend(self, event_id: str) -> dict:
        """
        Série temporal de estatísticas por minuto.
        Disponível para jogos a partir de 2017-06-10.
        """
        return self._get("/v1/event/stats_trend", {"event_id": event_id})

    def get_prematch_odds(self, event_id: str) -> dict:
        """Odds pré-jogo Bet365. Parâmetro FI = fixture/event ID."""
        return self._get("/v3/bet365/prematch", {"FI": event_id})

    def get_inplay_odds(self, event_id: str) -> dict:
        """Odds ao vivo Bet365 para um evento."""
        return self._get("/v1/bet365/inplay", {"FI": event_id})

    def get_h2h(self, event_id: str) -> dict:
        """Confrontos diretos (H2H) entre os dois times do evento."""
        return self._get("/v1/h2h", {"event_id": event_id})


# ---------------------------------------------------------------------------
# Parsers de resposta
# ---------------------------------------------------------------------------

def parse_score(score_str: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """'2-1' → (2, 1). Retorna (None, None) se inválido."""
    if not score_str or "-" not in str(score_str):
        return None, None
    try:
        parts = str(score_str).split("-")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None, None


def parse_stat_value(val) -> tuple[Optional[int], Optional[int]]:
    """
    Stats vêm no formato "home_val:away_val" (ex: "4:7") ou como int/float.
    Retorna (home, away).
    """
    if val is None:
        return None, None
    s = str(val)
    if ":" in s:
        try:
            h, a = s.split(":")
            return int(h), int(a)
        except ValueError:
            return None, None
    try:
        return int(val), None
    except (ValueError, TypeError):
        return None, None


def extract_event_metadata(event: dict) -> dict:
    """Extrai campos de identificação de um evento."""
    league = event.get("league", {}) or {}
    home   = event.get("home", {}) or {}
    away   = event.get("away", {}) or {}
    scores = event.get("ss", "")          # placar atual "home-away"
    timer  = event.get("timer", {}) or {}

    h_score, a_score = parse_score(scores)

    return {
        "event_id":    str(event.get("id", "")),
        "league_id":   str(league.get("id", "")),
        "league_name": league.get("name", ""),
        "home_team":   home.get("name", ""),
        "away_team":   away.get("name", ""),
        "home_id":     str(home.get("id", "")),
        "away_id":     str(away.get("id", "")),
        "time_unix":   event.get("time", None),
        "kickoff_dt":  datetime.utcfromtimestamp(int(event["time"])).isoformat()
                       if event.get("time") else None,
        "match_minute": timer.get("tm", None),
        "match_second": timer.get("ts", None),
        "score_home":  h_score,
        "score_away":  a_score,
    }


def parse_stats_trend(raw_trend: list, event_id: str, meta: dict,
                      source: str = "historico") -> list[dict]:
    """
    Converte a lista de snapshots do stats_trend em linhas de DataFrame.

    Cada item da lista `raw_trend` representa um minuto do jogo e contém
    pares de campos [home_val, away_val] por estatística.

    Campos mapeados (baseado na documentação BetsAPI):
      0  - corners
      1  - attacks
      2  - dangerous_attacks
      3  - on_target (chutes a gol)
      4  - off_target (chutes fora)
      5  - possession %
      6  - yellow_cards
      7  - red_cards
      8  - free_kicks / faltas
    """
    FIELD_MAP = {
        0: ("corners_home",           "corners_away"),
        1: ("attacks_home",           "attacks_away"),
        2: ("dangerous_attacks_home", "dangerous_attacks_away"),
        3: ("shots_on_target_home",   "shots_on_target_away"),
        4: ("shots_off_target_home",  "shots_off_target_away"),
        5: ("possession_home",        "possession_away"),
        6: ("yellow_cards_home",      "yellow_cards_away"),
        7: ("red_cards_home",         "red_cards_away"),
        8: ("fouls_home",             "fouls_away"),
        9: ("saves_home",             "saves_away"),
        10: ("offsides_home",         "offsides_away"),
        11: ("goal_kicks_home",       "goal_kicks_away"),
    }

    rows = []
    if not raw_trend:
        return rows

    for minute_idx, snapshot in enumerate(raw_trend):
        row = {
            "event_id":         event_id,
            "minute":           minute_idx + 1,
            "collection_source": source,
            "collected_at":     datetime.utcnow().isoformat(),
            **{k: v for k, v in meta.items()},
        }
        # snapshot pode ser lista de listas ou dict — BetsAPI retorna lista
        stats_list = snapshot if isinstance(snapshot, list) else []
        for field_idx, (home_col, away_col) in FIELD_MAP.items():
            if field_idx < len(stats_list):
                entry = stats_list[field_idx]
                if isinstance(entry, list) and len(entry) >= 2:
                    row[home_col] = _to_int(entry[0])
                    row[away_col] = _to_int(entry[1])
                elif isinstance(entry, str) and ":" in entry:
                    h, a = parse_stat_value(entry)
                    row[home_col] = h
                    row[away_col] = a
                else:
                    row[home_col] = None
                    row[away_col] = None
            else:
                row[home_col] = None
                row[away_col] = None

        rows.append(row)
    return rows


def _to_int(val) -> Optional[int]:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def build_panorama_row(event_id: str, meta: dict, event_view: dict,
                       snapshot_rows: list[dict],
                       source: str = "historico") -> dict:
    """
    Cria 1 linha de resumo por jogo para o panorama final.
    Inclui totais finais e médias por minuto (útil para features de ML).
    """
    result = event_view.get("results", [{}]) or [{}]
    result = result[0] if result else {}
    stats  = result.get("stats", {}) or {}

    # Placar final via /event/view (mais confiável que inplay)
    ss = result.get("ss", meta.get("score_home", ""))
    final_h, final_a = parse_score(ss)

    def s(key):
        h, a = parse_stat_value(stats.get(key))
        return h, a

    corners_h, corners_a = s("corners")
    throw_ins_h, throw_ins_a = s("throw_ins")
    tackles_h, tackles_a = s("tackles")
    ball_safe_h, ball_safe_a = s("ball_safe")

    row = {
        "event_id":         event_id,
        "league_id":        meta["league_id"],
        "league_name":      meta["league_name"],
        "home_team":        meta["home_team"],
        "home_id":          meta.get("home_id", ""),
        "away_team":        meta["away_team"],
        "away_id":          meta.get("away_id", ""),
        "kickoff_dt":       meta["kickoff_dt"],
        # Placar final
        "final_score_home": final_h,
        "final_score_away": final_a,
        "total_goals":      (final_h or 0) + (final_a or 0),
        # Escanteios totais (fonte: /event/view stats)
        "collection_source":              source,
        "collected_at":                   datetime.utcnow().isoformat(),
        "corners_home_total":             corners_h,
        "corners_away_total":             corners_a,
        "corners_total":                  (corners_h or 0) + (corners_a or 0),
        # Demais stats finais
        "possession_home_avg":            _safe_avg(snapshot_rows, "possession_home"),
        "possession_away_avg":            _safe_avg(snapshot_rows, "possession_away"),
        "shots_on_target_home_total":     _safe_last(snapshot_rows, "shots_on_target_home"),
        "shots_on_target_away_total":     _safe_last(snapshot_rows, "shots_on_target_away"),
        "shots_off_target_home_total":    _safe_last(snapshot_rows, "shots_off_target_home"),
        "shots_off_target_away_total":    _safe_last(snapshot_rows, "shots_off_target_away"),
        "attacks_home_total":             _safe_last(snapshot_rows, "attacks_home"),
        "attacks_away_total":             _safe_last(snapshot_rows, "attacks_away"),
        "dangerous_attacks_home_total":   _safe_last(snapshot_rows, "dangerous_attacks_home"),
        "dangerous_attacks_away_total":   _safe_last(snapshot_rows, "dangerous_attacks_away"),
        "yellow_cards_home_total":        _safe_last(snapshot_rows, "yellow_cards_home"),
        "yellow_cards_away_total":        _safe_last(snapshot_rows, "yellow_cards_away"),
        "red_cards_home_total":           _safe_last(snapshot_rows, "red_cards_home"),
        "red_cards_away_total":           _safe_last(snapshot_rows, "red_cards_away"),
        "fouls_home_total":               _safe_last(snapshot_rows, "fouls_home"),
        "fouls_away_total":               _safe_last(snapshot_rows, "fouls_away"),
        # Saves / Offsides / Goal kicks (índices 9-11 do stats_trend)
        "saves_home_total":               _safe_last(snapshot_rows, "saves_home"),
        "saves_away_total":               _safe_last(snapshot_rows, "saves_away"),
        "offsides_home_total":            _safe_last(snapshot_rows, "offsides_home"),
        "offsides_away_total":            _safe_last(snapshot_rows, "offsides_away"),
        "goal_kicks_home_total":          _safe_last(snapshot_rows, "goal_kicks_home"),
        "goal_kicks_away_total":          _safe_last(snapshot_rows, "goal_kicks_away"),
        # Stats adicionais do event/view (não disponíveis no stats_trend)
        "throw_ins_home_total":           throw_ins_h,
        "throw_ins_away_total":           throw_ins_a,
        "tackles_home_total":             tackles_h,
        "tackles_away_total":             tackles_a,
        "ball_safe_home_total":           ball_safe_h,
        "ball_safe_away_total":           ball_safe_a,
        # Placar do intervalo (via event/view scores["1"])
        "ht_score_home":                  _extract_ht_score(event_view)[0],
        "ht_score_away":                  _extract_ht_score(event_view)[1],
        # Escanteios por tempo (1º e 2º)
        "corners_home_ht":   _safe_last_filtered(snapshot_rows, "corners_home",  lambda m: m <= 45),
        "corners_away_ht":   _safe_last_filtered(snapshot_rows, "corners_away",  lambda m: m <= 45),
        "corners_ht_total":  (_safe_last_filtered(snapshot_rows, "corners_home", lambda m: m <= 45) or 0)
                             + (_safe_last_filtered(snapshot_rows, "corners_away", lambda m: m <= 45) or 0),
        "corners_home_2h":   (corners_h or 0) - (_safe_last_filtered(snapshot_rows, "corners_home", lambda m: m <= 45) or 0),
        "corners_away_2h":   (corners_a or 0) - (_safe_last_filtered(snapshot_rows, "corners_away", lambda m: m <= 45) or 0),
        "corners_2h_total":  (corners_h or 0) + (corners_a or 0)
                             - (_safe_last_filtered(snapshot_rows, "corners_home", lambda m: m <= 45) or 0)
                             - (_safe_last_filtered(snapshot_rows, "corners_away", lambda m: m <= 45) or 0),
        # Minuto do primeiro escanteio (útil para ML ao vivo)
        "first_corner_minute":            _first_corner_minute(snapshot_rows),
        "total_snapshot_minutes":         len(snapshot_rows),
    }
    return row


def _safe_avg(rows, col) -> Optional[float]:
    vals = [r[col] for r in rows if r.get(col) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _safe_last(rows, col) -> Optional[int]:
    """Pega o último valor não-nulo da série (valor acumulado ao final)."""
    for r in reversed(rows):
        if r.get(col) is not None:
            return r[col]
    return None


def _first_corner_minute(rows) -> Optional[int]:
    """Minuto em que aparece o primeiro escanteio (home ou away > 0)."""
    for r in rows:
        h = r.get("corners_home") or 0
        a = r.get("corners_away") or 0
        if h + a > 0:
            return r["minute"]
    return None


def _safe_last_filtered(rows, col, minute_filter_fn) -> Optional[int]:
    """Último valor não-nulo de `col` entre linhas que passam no filtro de minuto."""
    filtered = [r for r in rows if minute_filter_fn(r.get("minute", 0))]
    return _safe_last(filtered, col)


def _extract_ht_score(event_view: dict) -> tuple[Optional[int], Optional[int]]:
    """
    Extrai placar do intervalo do event_view.
    BetsAPI armazena scores por período: scores["1"]["home"] / ["away"]
    """
    try:
        result = (event_view.get("results", [{}]) or [{}])[0]
        scores = result.get("scores", {}) or {}
        ht = scores.get("1", {}) or {}
        return _to_int(ht.get("home")), _to_int(ht.get("away"))
    except (IndexError, AttributeError, TypeError):
        return None, None


def parse_prematch_odds(odds_resp: dict) -> dict:
    """
    Extrai odds pré-jogo da resposta /v3/bet365/prematch.

    Mercados extraídos:
      - corners (Over/Under) + asian_corners (Home/Away)
      - match_result / full_time_result (1x2)
      - goals_over_under (Goals O/U)
      - both_teams_to_score (BTTS)

    Retorna dict com todas as chaves (None se dados indisponíveis).
    """
    result = {
        # Escanteios
        "corners_line":            None,
        "corners_over_odds":       None,
        "corners_under_odds":      None,
        "asian_corners_line":      None,
        "asian_corners_home_odds": None,
        "asian_corners_away_odds": None,
        # Resultado 1x2
        "odds_home_win":           None,
        "odds_draw":               None,
        "odds_away_win":           None,
        # Gols Over/Under
        "goals_line":              None,
        "goals_over_odds":         None,
        "goals_under_odds":        None,
        # BTTS (ambas marcam)
        "btts_yes_odds":           None,
        "btts_no_odds":            None,
    }
    try:
        sp = (
            (odds_resp.get("results", [{}]) or [{}])[0]
            .get("main", {})
            .get("sp", {})
        )

        # --- Escanteios Over/Under ---
        for entry in (sp.get("corners", {}).get("odds", []) or []):
            name = str(entry.get("name", "")).lower()
            header, odds_v = entry.get("header"), entry.get("odds")
            if name == "over" and odds_v is not None:
                result["corners_over_odds"] = float(odds_v)
                result["corners_line"] = float(header) if header else None
            elif name == "under" and odds_v is not None:
                result["corners_under_odds"] = float(odds_v)

        # --- Asian Corners ---
        for entry in (sp.get("asian_corners", {}).get("odds", []) or []):
            name = str(entry.get("name", "")).lower()
            header, odds_v = entry.get("header"), entry.get("odds")
            if name == "home" and odds_v is not None:
                result["asian_corners_home_odds"] = float(odds_v)
                result["asian_corners_line"] = float(header) if header else None
            elif name == "away" and odds_v is not None:
                result["asian_corners_away_odds"] = float(odds_v)

        # --- Resultado 1x2 ---
        # BetsAPI pode usar diferentes chaves para o mercado 1x2
        match_sp = sp.get("match_result") or sp.get("full_time_result") or {}
        for entry in (match_sp.get("odds", []) or []):
            name = str(entry.get("name", "")).strip()
            odds_v = entry.get("odds")
            if odds_v is None:
                continue
            if name in ("1", "Home"):
                result["odds_home_win"] = float(odds_v)
            elif name in ("X", "Draw"):
                result["odds_draw"] = float(odds_v)
            elif name in ("2", "Away"):
                result["odds_away_win"] = float(odds_v)

        # --- Gols Over/Under ---
        goals_sp = sp.get("goals_over_under") or sp.get("totals") or {}
        for entry in (goals_sp.get("odds", []) or []):
            name = str(entry.get("name", "")).lower()
            header, odds_v = entry.get("header"), entry.get("odds")
            if name == "over" and odds_v is not None:
                result["goals_over_odds"] = float(odds_v)
                result["goals_line"] = float(header) if header else None
            elif name == "under" and odds_v is not None:
                result["goals_under_odds"] = float(odds_v)

        # --- BTTS (ambas marcam) ---
        btts_sp = sp.get("both_teams_to_score") or sp.get("btts") or {}
        for entry in (btts_sp.get("odds", []) or []):
            name = str(entry.get("name", "")).lower()
            odds_v = entry.get("odds")
            if odds_v is None:
                continue
            if name in ("yes", "sim"):
                result["btts_yes_odds"] = float(odds_v)
            elif name in ("no", "não", "nao"):
                result["btts_no_odds"] = float(odds_v)

    except (IndexError, AttributeError, TypeError, KeyError, ValueError):
        pass
    return result


def parse_inplay_corner_odds(odds_resp: dict) -> dict:
    """
    Extrai odds ao vivo de escanteios da resposta /v1/bet365/inplay.

    Retorna a linha e odds O/U de escanteios ao vivo (None se indisponíveis).
    """
    result = {
        "live_corners_line":       None,
        "live_corners_over_odds":  None,
        "live_corners_under_odds": None,
    }
    try:
        results = odds_resp.get("results", []) or []
        for market in results:
            market_name = str(market.get("name", "")).lower()
            if "corner" not in market_name:
                continue
            if "over" in market_name or "total" in market_name:
                for entry in (market.get("odds", []) or []):
                    name = str(entry.get("name", "")).lower()
                    header = entry.get("header")
                    odds_v = entry.get("odds")
                    if name == "over" and odds_v is not None:
                        result["live_corners_over_odds"] = float(odds_v)
                        result["live_corners_line"] = float(header) if header else None
                    elif name == "under" and odds_v is not None:
                        result["live_corners_under_odds"] = float(odds_v)
                if result["live_corners_line"] is not None:
                    break
    except (IndexError, AttributeError, TypeError, KeyError, ValueError):
        pass
    return result


def parse_h2h_corners(h2h_resp: dict) -> dict:
    """
    Extrai estatísticas de escanteios dos confrontos diretos (H2H).

    Analisa os últimos jogos entre os dois times e calcula médias
    de escanteios. Retorna dict com métricas H2H.
    """
    result = {
        "h2h_total_games":         0,
        "h2h_avg_corners_total":   None,
        "h2h_avg_corners_home":    None,
        "h2h_avg_corners_away":    None,
        "h2h_avg_goals_total":     None,
    }
    try:
        events = h2h_resp.get("results", {}).get("h2h", []) or []
        if not events:
            return result

        corner_totals = []
        corner_homes  = []
        corner_aways  = []
        goal_totals   = []

        for ev in events:
            stats = ev.get("stats", {}) or {}
            corners_str = stats.get("corners")
            if corners_str:
                ch, ca = parse_stat_value(corners_str)
                if ch is not None and ca is not None:
                    corner_totals.append(ch + ca)
                    corner_homes.append(ch)
                    corner_aways.append(ca)

            ss = ev.get("ss", "")
            gh, ga = parse_score(ss)
            if gh is not None and ga is not None:
                goal_totals.append(gh + ga)

        result["h2h_total_games"] = len(events)
        if corner_totals:
            result["h2h_avg_corners_total"] = round(sum(corner_totals) / len(corner_totals), 2)
            result["h2h_avg_corners_home"]  = round(sum(corner_homes)  / len(corner_homes),  2)
            result["h2h_avg_corners_away"]  = round(sum(corner_aways)  / len(corner_aways),  2)
        if goal_totals:
            result["h2h_avg_goals_total"] = round(sum(goal_totals) / len(goal_totals), 2)

    except (AttributeError, TypeError, KeyError, ValueError):
        pass
    return result


# ---------------------------------------------------------------------------
# Checkpoint — salva e retoma progresso entre execuções
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Persiste o progresso da coleta histórica em um arquivo JSON.
    Permite retomar de onde parou após atingir o rate limit ou interrupção.
    """

    def __init__(self, path: Path):
        self.path = path

    def save(self, data: dict):
        data["saved_at"] = datetime.utcnow().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Checkpoint salvo → %s", self.path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("Checkpoint carregado: dia=%s página=%d eventos=%d",
                 data.get("current_day"), data.get("current_page", 1),
                 data.get("total_events", 0))
        return data

    def clear(self):
        if self.path.exists():
            self.path.unlink()
            log.info("Checkpoint removido (coleta concluída).")

    def exists(self) -> bool:
        return self.path.exists()


# ---------------------------------------------------------------------------
# Salvamento incremental
# ---------------------------------------------------------------------------

class DataSaver:
    """
    Mantém dois buffers em memória e salva em Parquet + CSV.
    Ao gravar, mescla com dados já existentes no disco e deduplica por event_id,
    garantindo segurança em caso de retomada com --resume.
    """

    def __init__(self, output_dir: str, save_csv: bool = True):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.save_csv = save_csv
        self.snapshots: list[dict] = []
        self.panoramas: list[dict] = []

    def add_snapshots(self, rows: list[dict]):
        self.snapshots.extend(rows)

    def add_panorama(self, row: dict):
        self.panoramas.append(row)

    def flush(self):
        """Grava tudo no disco mesclando com dados existentes (safe para --resume)."""
        self._save_df(self.snapshots, "snapshots_por_minuto",
                      dedup_cols=["event_id", "minute"])
        self._save_df(self.panoramas, "panorama_jogos",
                      dedup_cols=["event_id"])
        log.info("Dados salvos → %s (%d snapshots em buffer, %d jogos em buffer)",
                 self.out, len(self.snapshots), len(self.panoramas))

    def _save_df(self, data: list[dict], name: str, dedup_cols: list[str]):
        if not data:
            return
        df_new = pd.DataFrame(data)
        pq_path = self.out / f"{name}.parquet"

        # Mescla com dados existentes e deduplica (mantém o mais recente)
        if pq_path.exists():
            try:
                df_existing = pd.read_parquet(pq_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                valid_dedup = [c for c in dedup_cols if c in df_combined.columns]
                if valid_dedup:
                    # Deduplicação inteligente: prefere "historico" sobre "live"
                    # quando o mesmo event_id existir em ambas as origens
                    if "collection_source" in df_combined.columns:
                        source_order = {"live": 0, "historico": 1}
                        df_combined["_src_order"] = (
                            df_combined["collection_source"]
                            .map(source_order)
                            .fillna(0)
                        )
                        df_combined = df_combined.sort_values("_src_order")
                        df_combined = df_combined.drop(columns=["_src_order"])
                    df_combined = df_combined.drop_duplicates(
                        subset=valid_dedup, keep="last"
                    ).reset_index(drop=True)
                df_new = df_combined
            except Exception as exc:
                log.warning("Erro ao mesclar parquet existente (%s): %s — sobrescrevendo.", name, exc)

        df_new.to_parquet(pq_path, index=False)
        if self.save_csv:
            df_new.to_csv(self.out / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# Modo HISTÓRICO
# ---------------------------------------------------------------------------

def _fmt_time(seconds: int) -> str:
    """Formata segundos em string legível: '1h 23m' ou '45m 10s'."""
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def run_historico(
    client: BetsAPIClient,
    saver: DataSaver,
    start_day: str,
    end_day: str,
    league_id: Optional[int] = None,
    flush_every: int = 50,
    checkpoint: Optional[CheckpointManager] = None,
    resume: bool = False,
    max_games: Optional[int] = None,
):
    """
    Itera de start_day até end_day (inclusive), coleta todos os jogos
    finalizados de futebol, seus stats_trend e panorama.

    start_day / end_day: formato YYYYMMDD

    Comportamentos:
      - Novo job       : apaga checkpoint antigo, carrega collected_ids do Parquet
      - --resume       : retoma do dia salvo no checkpoint, skipa jogos já coletados
      - auto_wait=True : pausa ao atingir rate limit, retoma automaticamente
      - auto_wait=False: salva checkpoint e encerra ao atingir rate limit
      - Ctrl+C         : salva checkpoint do DIA ATUAL e encerra
    """
    start = datetime.strptime(start_day, "%Y%m%d")
    end   = datetime.strptime(end_day,   "%Y%m%d")
    days  = [(start + timedelta(days=i)).strftime("%Y%m%d")
             for i in range((end - start).days + 1)]

    # --- Carrega checkpoint se solicitado ---
    resume_day = None
    if resume and checkpoint and checkpoint.exists():
        ckpt_data  = checkpoint.load()
        resume_day = ckpt_data.get("current_day")

    # --- Carrega event_ids já coletados (sempre ativo) ---
    collected_ids: set[str] = set()
    pano_path = saver.out / "panorama_jogos.parquet"
    if pano_path.exists():
        try:
            df_ex = pd.read_parquet(pano_path, columns=["event_id"])
            collected_ids = set(df_ex["event_id"].astype(str))
        except Exception as exc:
            log.warning("Erro ao carregar IDs existentes: %s", exc)

    # total_events reflete o estado real (jogos já no Parquet + novos desta sessão)
    total_events   = len(collected_ids)
    session_new    = 0   # novos coletados nesta sessão
    session_skip   = 0   # pulados nesta sessão

    # --- Banner de início ---
    days_pending = sum(1 for d in days if not resume_day or d >= resume_day)
    print(f"\n{'═'*62}")
    print(f"  BetsAPI Corner Collector  —  {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'─'*62}")
    print(f"  Range     : {start_day} → {end_day}  ({len(days)} dias)")
    if resume_day:
        print(f"  Retomando : a partir de {resume_day}  ({days_pending} dias restantes)")
    print(f"  Limite    : {client.max_requests} req/hora  |  "
          f"Auto-wait: {'ON ✓' if client.auto_wait else 'OFF'}")
    if collected_ids:
        print(f"  No Parquet: {len(collected_ids):,} jogos já coletados  →  serão pulados")
    if league_id:
        print(f"  Liga      : {league_id}")
    print(f"{'═'*62}\n")

    h2h_cache:  dict[tuple[str, str], dict] = {}
    current_day = days[0] if days else start_day  # BUG FIX: rastreia dia atual para Ctrl+C

    def _save_checkpoint(day: str):
        if checkpoint:
            checkpoint.save({
                "start_day":    start_day,
                "end_day":      end_day,
                "current_day":  day,
                "total_events": total_events,
                "league_id":    league_id,
            })
            print(f"\n  💾 Checkpoint salvo — dia {day}")
            print(f"  ▶  Para retomar: adicione --resume ao comando\n")

    try:
        for day_idx, day in enumerate(days, 1):
            current_day = day  # BUG FIX: atualiza sempre antes de qualquer operação

            # Pula dias completamente anteriores ao ponto de retomada
            if resume_day and day < resume_day:
                continue

            # --- Cabeçalho do dia ---
            day_label = datetime.strptime(day, "%Y%m%d").strftime("%d/%m/%Y")
            days_done = sum(1 for d in days[:day_idx] if not resume_day or d >= resume_day)
            print(f"┌─ [{days_done:>3}/{days_pending}] {day_label} "
                  f"{'─' * (44 - len(day_label))}")

            page          = 1
            day_new       = 0
            day_skipped   = 0
            day_requests  = client.request_count

            while True:
                try:
                    resp = client.get_ended_events(day, page=page, league_id=league_id)
                    time.sleep(REQUEST_DELAY)
                except RateLimitReached as exc:
                    log.warning("Rate limit (auto_wait=False): %s", exc)
                    _save_checkpoint(day)
                    saver.flush()
                    return

                events = resp.get("results", []) or []
                pager  = resp.get("pager", {}) or {}
                total_pages = int(pager.get("total_pages", 1))

                if not events:
                    print(f"│  Pág {page}: sem jogos")
                    break

                page_new     = 0
                page_skipped = 0

                for event in events:
                    event_id = str(event.get("id", ""))
                    if not event_id:
                        continue

                    # Pula jogos já coletados — zero requests extras
                    if event_id in collected_ids:
                        page_skipped += 1
                        day_skipped  += 1
                        session_skip += 1
                        continue

                    meta = extract_event_metadata(event)
                    home = meta.get("home_team", "?")
                    away = meta.get("away_team", "?")

                    try:
                        # Stats trend
                        trend_resp = client.get_stats_trend(event_id)
                        time.sleep(REQUEST_DELAY)
                        raw_trend     = trend_resp.get("results", []) or []
                        snapshot_rows = parse_stats_trend(raw_trend, event_id, meta,
                                                          source="historico")

                        # Event view
                        view_resp   = client.get_event_view(event_id)
                        time.sleep(REQUEST_DELAY)
                        panorama_row = build_panorama_row(event_id, meta, view_resp,
                                                          snapshot_rows, source="historico")

                        # H2H (com cache)
                        pair_key = (meta["home_id"], meta["away_id"])
                        if pair_key not in h2h_cache:
                            h2h_resp = client.get_h2h(event_id)
                            time.sleep(REQUEST_DELAY)
                            h2h_cache[pair_key] = parse_h2h_corners(h2h_resp)
                        panorama_row.update(h2h_cache[pair_key])

                    except RateLimitReached as exc:
                        # Só ocorre se auto_wait=False
                        log.warning("Rate limit ao processar %s: %s", event_id, exc)
                        _save_checkpoint(day)
                        saver.flush()
                        return

                    saver.add_snapshots(snapshot_rows)
                    saver.add_panorama(panorama_row)
                    collected_ids.add(event_id)
                    total_events  += 1
                    session_new   += 1
                    page_new      += 1
                    day_new       += 1

                    if total_events % flush_every == 0:
                        saver.flush()

                    # Limite de jogos para testes
                    if max_games and session_new >= max_games:
                        print(f"\n  🏁 --max-games {max_games} atingido — encerrando.")
                        saver.flush()
                        _print_session_summary(session_new, session_skip, total_events, client)
                        return

                # --- Status da página ---
                req_used = client.request_count
                req_pct  = req_used / client.max_requests * 100
                status   = (f"│  Pág {page}/{total_pages}: "
                            f"{len(events)} jogos  |  "
                            f"✦ {page_new} novos  "
                            f"✓ {page_skipped} pulados  |  "
                            f"req: {req_used}/{client.max_requests} ({req_pct:.0f}%)")
                print(status)

                if page >= total_pages:
                    break
                page += 1

            # --- Resumo do dia ---
            # max(0,...) protege contra reset de janela que zera request_count
            day_req_used = client.request_count - day_requests
            if day_req_used < 0:
                day_req_used += client.max_requests  # janela resetou durante o dia
            print(f"└─ {day_new} coletados  {day_skipped} pulados  "
                  f"{day_req_used} requests neste dia\n")

            if day_new > 0 or day_skipped == 0:
                saver.flush()

    except KeyboardInterrupt:
        print(f"\n\n  ⚠  Interrompido pelo usuário (Ctrl+C).")
        print(f"  Dia atual: {current_day}")
        _save_checkpoint(current_day)  # BUG FIX: usa current_day rastreado, não days[-1]
        saver.flush()
        _print_session_summary(session_new, session_skip, total_events, client)
        return

    saver.flush()
    if checkpoint:
        checkpoint.clear()

    # --- Resumo final ---
    print(f"\n{'═'*62}")
    print(f"  ✅ COLETA CONCLUÍDA")
    _print_session_summary(session_new, session_skip, total_events, client)
    print(f"{'═'*62}\n")


def _print_session_summary(new: int, skipped: int, total: int, client: BetsAPIClient):
    """Imprime resumo da sessão de coleta."""
    print(f"{'─'*62}")
    print(f"  Esta sessão : {new:>6,} novos coletados  |  {skipped:>6,} pulados")
    print(f"  Total Parquet: {total:>6,} jogos")
    print(f"  Requests    : {client.request_count:>6,} / {client.max_requests:,} usados")


# ---------------------------------------------------------------------------
# Modo AO VIVO
# ---------------------------------------------------------------------------

def run_live(
    client: BetsAPIClient,
    saver: DataSaver,
    interval: int = 60,
    max_iterations: Optional[int] = None,
):
    """
    Loop contínuo que:
      1. Busca todos os jogos ao vivo (/v3/events/inplay)
      2. Para cada jogo, coleta stats_trend do minuto atual
      3. Aguarda `interval` segundos antes do próximo ciclo
      4. Ao detectar que um jogo terminou, coleta o panorama final

    max_iterations: útil para testes (None = loop infinito)
    """
    log.info("Iniciando modo AO VIVO (intervalo: %ds). Ctrl+C para encerrar.", interval)
    active_events: dict[str, dict] = {}  # event_id → meta
    iteration = 0

    try:
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1
            log.info("--- Ciclo %d | %s ---", iteration, datetime.utcnow().isoformat())

            inplay_resp = client.get_inplay_events()
            time.sleep(REQUEST_DELAY)
            live_events = inplay_resp.get("results", []) or []
            live_ids = set()

            for event in live_events:
                event_id = str(event.get("id", ""))
                if not event_id:
                    continue
                live_ids.add(event_id)
                meta = extract_event_metadata(event)
                active_events[event_id] = meta

                # Snapshot do minuto atual
                trend_resp = client.get_stats_trend(event_id)
                time.sleep(REQUEST_DELAY)
                raw_trend = trend_resp.get("results", []) or []
                snapshot_rows = parse_stats_trend(raw_trend, event_id, meta, source="live")

                if snapshot_rows:
                    latest = snapshot_rows[-1].copy()
                    saver.add_snapshots([latest])

            # Detecta jogos que saíram do ao vivo (terminaram)
            finished = set(active_events.keys()) - live_ids
            for event_id in finished:
                meta = active_events.pop(event_id)
                log.info("Jogo finalizado detectado: %s vs %s (id=%s)",
                         meta["home_team"], meta["away_team"], event_id)

                # Coleta panorama final
                trend_resp = client.get_stats_trend(event_id)
                time.sleep(REQUEST_DELAY)
                raw_trend = trend_resp.get("results", []) or []
                all_snapshots = parse_stats_trend(raw_trend, event_id, meta, source="live")

                view_resp = client.get_event_view(event_id)
                time.sleep(REQUEST_DELAY)
                panorama_row = build_panorama_row(event_id, meta, view_resp, all_snapshots,
                                                  source="live")

                # H2H para o panorama de jogos finalizados ao vivo
                h2h_resp = client.get_h2h(event_id)
                time.sleep(REQUEST_DELAY)
                panorama_row.update(parse_h2h_corners(h2h_resp))

                saver.add_panorama(panorama_row)

            saver.flush()
            log.info("Jogos ao vivo: %d | Finalizados neste ciclo: %d", len(live_ids), len(finished))

            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Interrompido pelo usuário.")
        saver.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Coletor de dados de escanteios via BetsAPI"
    )
    parser.add_argument(
        "--token", required=True,
        help="Token de autenticação da BetsAPI"
    )
    parser.add_argument(
        "--mode", choices=["historico", "live"], default="historico",
        help="Modo de operação: 'historico' ou 'live'"
    )
    parser.add_argument(
        "--output", default="dados_escanteios",
        help="Diretório de saída para os arquivos (padrão: dados_escanteios)"
    )
    # Histórico
    parser.add_argument(
        "--start", default=None,
        help="Data início YYYYMMDD (modo historico). Padrão: 7 dias atrás"
    )
    parser.add_argument(
        "--end", default=None,
        help="Data fim YYYYMMDD (modo historico). Padrão: ontem"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Atalho: coletar os últimos N dias (ignora --start/--end)"
    )
    parser.add_argument(
        "--league-id", type=int, default=None,
        help="Filtrar por league_id específico (opcional)"
    )
    # Live
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Intervalo em segundos entre ciclos no modo live (padrão: 60)"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Não gerar cópias CSV (apenas Parquet)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Retoma coleta histórica de onde parou (requer checkpoint existente)"
    )
    parser.add_argument(
        "--max-requests", type=int, default=3500,
        help="Limite de requests por hora (padrão: 3500, plano Soccer API tem 3600)"
    )
    parser.add_argument(
        "--no-auto-wait", action="store_true",
        help="Desativa pausa automática ao atingir rate limit — encerra e salva checkpoint"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Para após coletar N jogos novos (útil para testes rápidos)"
    )
    parser.add_argument(
        "--exclude-esports", action="store_true",
        help="Pula ligas de esports/virtual/fantasy (ativa filtro automático)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    client = BetsAPIClient(
        token=args.token,
        max_requests=args.max_requests,
        auto_wait=not args.no_auto_wait,
    )
    saver  = DataSaver(output_dir=args.output, save_csv=not args.no_csv)

    if args.mode == "historico":
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointManager(output_path / "checkpoint.json")

        today = datetime.utcnow()

        # Se --resume, lê datas do checkpoint (ignora --start/--end/--days)
        if args.resume and checkpoint.exists():
            ckpt = checkpoint.load()
            start_day = ckpt.get("start_day")
            end_day   = ckpt.get("end_day")
            if not start_day or not end_day:
                log.error("Checkpoint inválido — sem start_day/end_day. Use sem --resume.")
                return
            log.info("Datas lidas do checkpoint: %s → %s", start_day, end_day)
        else:
            if args.days:
                end_day   = (today - timedelta(days=1)).strftime("%Y%m%d")
                start_day = (today - timedelta(days=args.days)).strftime("%Y%m%d")
            else:
                end_day   = args.end   or (today - timedelta(days=1)).strftime("%Y%m%d")
                start_day = args.start or (today - timedelta(days=7)).strftime("%Y%m%d")
            # Novo job sem --resume: apaga checkpoint antigo automaticamente
            if checkpoint.exists():
                log.info("Novo job iniciado — checkpoint anterior removido.")
                checkpoint.clear()

        run_historico(
            client=client,
            saver=saver,
            start_day=start_day,
            end_day=end_day,
            league_id=args.league_id,
            checkpoint=checkpoint,
            resume=args.resume,
            max_games=args.max_games,
        )

    elif args.mode == "live":
        run_live(
            client=client,
            saver=saver,
            interval=args.interval,
        )


if __name__ == "__main__":
    main()
