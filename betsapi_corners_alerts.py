"""
=============================================================================
BetsAPI – Alertas de Value Bet em Escanteios
=============================================================================
Filtra previsões do predictor e exibe alertas coloridos no terminal
apenas para jogos com edge significativo.

Níveis de alerta:
  🟢 FORTE  : edge >= --strong-edge (padrão 2.0)
  🟡 MODERADO: edge >= --min-edge   (padrão 1.0)

Uso:
    python betsapi_corners_alerts.py --token SEU_TOKEN
    python betsapi_corners_alerts.py --token SEU_TOKEN --interval 90 --min-edge 1.0
    python betsapi_corners_alerts.py --token SEU_TOKEN --leagues 1,2,3 --sound
=============================================================================
"""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from betsapi_corners_collector import BetsAPIClient
from betsapi_corners_predictor import (
    HistoryCache,
    load_history_cache,
    load_model,
    run_predictor_cycle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path("dados_escanteios")
ALERTS_PATH = DATA_DIR / "alertas_live.csv"

# ---------------------------------------------------------------------------
# Códigos ANSI
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Aplica sequências ANSI ao texto."""
    return "".join(codes) + str(text) + _RESET


# ---------------------------------------------------------------------------
# Formatação de alertas
# ---------------------------------------------------------------------------

def _edge_level(edge: Optional[float], strong_threshold: float) -> Optional[str]:
    """Retorna 'STRONG', 'MODERATE' ou None conforme o edge."""
    if edge is None:
        return None
    abs_e = abs(edge)
    if abs_e >= strong_threshold:
        return "STRONG"
    return "MODERATE"


def format_alert(pred: dict, strong_threshold: float) -> str:
    """Retorna string formatada com cores ANSI para um alerta."""
    level = _edge_level(pred.get("edge"), strong_threshold)
    edge  = pred.get("edge") or 0.0
    bet   = pred.get("bet") or "?"

    if level == "STRONG":
        icon   = "🟢"
        color  = _GREEN
        label  = "FORTE"
    else:
        icon   = "🟡"
        color  = _YELLOW
        label  = "MODERADO"

    score = (f"{pred.get('score_home')}-{pred.get('score_away')}"
             if pred.get("score_home") is not None else "?-?")
    linha = f"{pred['corners_line']:.1f}" if pred.get("corners_line") else "N/A"
    odds_str = ""
    if bet == "OVER" and pred.get("over_odds"):
        odds_str = f" @ {pred['over_odds']:.2f}"
    elif bet == "UNDER" and pred.get("under_odds"):
        odds_str = f" @ {pred['under_odds']:.2f}"

    match_str = f"{pred.get('home_team', '?')} vs {pred.get('away_team', '?')}"
    league    = pred.get("league_name", "")

    line1 = (
        f"{icon}  {_c(label, _BOLD, color)}  "
        f"{_c(match_str, _BOLD)} "
        f"{_c(f'({league})', _DIM)}"
    )
    line2 = (
        f"   Min: {_c(pred.get('minute', '?'), _CYAN)}  "
        f"Placar: {score}  "
        f"Corners: {pred.get('corners_now', 0)}  "
        f"Prev: {_c(pred.get('pred_total', '?'), _BOLD)}  "
        f"Restam: {pred.get('remaining', '?')}"
    )
    line3 = (
        f"   Linha: {linha}  "
        f"Edge: {_c(f'{edge:+.2f}', _BOLD, color)}  "
        f"Aposta: {_c(bet + odds_str, _BOLD, color)}"
    )
    return "\n".join([line1, line2, line3])


def format_cycle_header(cycle: int, n_live: int, n_edge: int, n_strong: int,
                         min_edge: float) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sep = _c("─" * 70, _DIM)
    header = (
        f"{sep}\n"
        f"  {_c('Ciclo ' + str(cycle), _BOLD)}  |  {now} UTC  |  "
        f"min-edge={min_edge}\n"
        f"  Jogos ao vivo: {_c(n_live, _CYAN)}  |  "
        f"Com edge: {_c(n_edge, _YELLOW)}  |  "
        f"Alertas fortes: {_c(n_strong, _GREEN)}\n"
        f"{sep}"
    )
    return header


# ---------------------------------------------------------------------------
# Salvamento de alertas
# ---------------------------------------------------------------------------

def save_alerts(alerts: list[dict], path: Path) -> None:
    if not alerts:
        return
    fieldnames = ["timestamp", "event_id", "league_name", "league_id",
                  "home_team", "away_team", "minute", "score_home", "score_away",
                  "corners_now", "pred_total", "remaining",
                  "corners_line", "edge", "bet", "over_odds", "under_odds", "alert_level"]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(alerts)


# ---------------------------------------------------------------------------
# Loop de alertas
# ---------------------------------------------------------------------------

def run_alerts_cycle(
    client: BetsAPIClient,
    model,
    hist_cache: HistoryCache,
    min_edge: float,
    strong_threshold: float,
    leagues_filter: Optional[set[str]],
    cycle: int,
    sound: bool,
) -> tuple[list[dict], int]:
    """
    Executa um ciclo de previsão, filtra por edge e ligas, exibe alertas.
    Retorna (alertas, n_live_total).
    """
    # Busca todas as previsões (sem filtro de min_edge aqui — filtramos manualmente)
    all_preds = run_predictor_cycle(client, model, hist_cache, min_edge=0.0)
    n_live = len(all_preds)

    # Filtra por liga
    if leagues_filter:
        all_preds = [p for p in all_preds
                     if str(p.get("league_id", "")) in leagues_filter]

    # Separa por nível de edge
    strong   = [p for p in all_preds
                if p.get("edge") is not None and abs(p["edge"]) >= strong_threshold]
    moderate = [p for p in all_preds
                if p.get("edge") is not None
                and min_edge <= abs(p["edge"]) < strong_threshold]
    n_edge   = len(strong) + len(moderate)

    # Header do ciclo
    print(format_cycle_header(cycle, n_live, n_edge, len(strong), min_edge))

    alerts_to_save: list[dict] = []

    for pred in strong + moderate:
        level = _edge_level(pred.get("edge"), strong_threshold)
        print(format_alert(pred, strong_threshold))
        print()

        rec = dict(pred)
        rec["alert_level"] = level
        alerts_to_save.append(rec)

    if not (strong + moderate):
        print(f"  {_c('Nenhum alerta neste ciclo.', _DIM)}")

    if sound and strong:
        _beep(len(strong))

    return alerts_to_save, n_live


def _beep(n: int = 1) -> None:
    """Emite N beeps no terminal (funciona em Windows e Unix)."""
    try:
        import sys
        for _ in range(n):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.15)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alertas de value bet em escanteios (BetsAPI)")
    p.add_argument("--token",       required=True, help="Token da BetsAPI")
    p.add_argument("--interval",    type=int,   default=90,  help="Intervalo entre ciclos (s)")
    p.add_argument("--min-edge",    type=float, default=1.0, help="Edge mínimo para alerta moderado")
    p.add_argument("--strong-edge", type=float, default=2.0, help="Edge mínimo para alerta forte")
    p.add_argument("--window",      type=int,   default=10,  help="Rolling window do histórico")
    p.add_argument("--leagues",     default=None,
                   help="Filtrar por IDs de liga (separados por vírgula, ex: 1,2,3)")
    p.add_argument("--sound",       action="store_true",
                   help="Beep no terminal ao detectar alerta forte")
    p.add_argument("--output",      default=str(ALERTS_PATH), help="CSV de saída")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    leagues_filter: Optional[set[str]] = None
    if args.leagues:
        leagues_filter = {lid.strip() for lid in args.leagues.split(",")}
        log.info("Filtrando ligas: %s", leagues_filter)

    client = BetsAPIClient(token=args.token)
    model  = load_model()
    cache  = load_history_cache(window=args.window)
    out    = Path(args.output)

    cycle = 0
    log.info(
        "Iniciando alertas ao vivo (intervalo=%ds, min-edge=%.1f, strong-edge=%.1f). "
        "Ctrl+C para parar.",
        args.interval, args.min_edge, args.strong_edge,
    )

    try:
        while True:
            cycle += 1
            alerts, n_live = run_alerts_cycle(
                client=client,
                model=model,
                hist_cache=cache,
                min_edge=args.min_edge,
                strong_threshold=args.strong_edge,
                leagues_filter=leagues_filter,
                cycle=cycle,
                sound=args.sound,
            )
            save_alerts(alerts, out)
            if alerts:
                log.info("Ciclo %d: %d alerta(s) salvo(s) em %s", cycle, len(alerts), out)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("Encerrado pelo usuário.")


if __name__ == "__main__":
    main()
