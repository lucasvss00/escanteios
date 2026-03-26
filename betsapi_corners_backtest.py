"""
=============================================================================
BetsAPI – Backtesting e Simulação de ROI
=============================================================================
Simula apostas em jogos históricos usando o modelo treinado e avalia
o desempenho em termos de win rate, ROI e profit acumulado.

Uso:
    python betsapi_corners_backtest.py
    python betsapi_corners_backtest.py --edge 0.5 --odds 1.85 --plot
    python betsapi_corners_backtest.py --edge 1.0 --snapshot 45
=============================================================================
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from betsapi_corners_predictor import FEATURE_COLS, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR       = Path("dados_escanteios")
FEATURES_PATH  = DATA_DIR / "features_ml.parquet"
REPORT_PATH    = DATA_DIR / "backtest_report.csv"

EDGE_BANDS = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float("inf"))]


# ---------------------------------------------------------------------------
# Simulação de apostas
# ---------------------------------------------------------------------------

def simulate_bets(
    df: pd.DataFrame,
    model,
    edge_threshold: float,
    odds_value: float,
    snapshot_filter: Optional[int] = None,
) -> pd.DataFrame:
    """
    Para cada linha do dataset (jogo × minuto), executa a previsão e decide
    se aposta OVER ou UNDER com base no edge vs. a linha de escanteios.

    Retorna DataFrame com colunas adicionais:
        pred, edge, bet_direction, bet_placed, outcome, profit
    """
    df = df.copy()

    if snapshot_filter is not None:
        df = df[df["snap_minute"] == snapshot_filter].copy()
        log.info("Filtrado para snap_minute=%d: %d amostras", snapshot_filter, len(df))

    # Verifica features disponíveis
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("%d features ausentes (serão NaN): %s...", len(missing), missing[:5])

    # Adiciona features ausentes como NaN
    for col in missing:
        df[col] = np.nan

    X = df[FEATURE_COLS].astype(float)
    df["pred"] = model.predict(X)

    # Filtra apenas jogos com linha de odds (sem ela não dá para apostar)
    df_bet = df[df["corners_line"].notna() & df["target_corners_total"].notna()].copy()
    log.info("Amostras com linha de odds: %d / %d", len(df_bet), len(df))

    df_bet["edge"] = df_bet["pred"] - df_bet["corners_line"]

    # Decisão de aposta
    df_bet["bet_placed"]    = abs(df_bet["edge"]) >= edge_threshold
    df_bet["bet_direction"] = np.where(df_bet["edge"] > 0, "OVER", "UNDER")

    # Resultado da aposta
    actual   = df_bet["target_corners_total"]
    line     = df_bet["corners_line"]
    over_win  = actual > line
    under_win = actual < line

    df_bet["outcome"] = np.where(
        ~df_bet["bet_placed"], "NO_BET",
        np.where(
            df_bet["bet_direction"] == "OVER",
            np.where(over_win,  "WIN", np.where(actual == line, "PUSH", "LOSS")),
            np.where(under_win, "WIN", np.where(actual == line, "PUSH", "LOSS")),
        )
    )

    # P&L por unidade apostada
    df_bet["profit"] = np.where(
        df_bet["outcome"] == "WIN",  odds_value - 1.0,
        np.where(df_bet["outcome"] == "PUSH",  0.0,
        np.where(df_bet["outcome"] == "LOSS", -1.0, 0.0))
    )

    return df_bet


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------

def _metrics_for(df_bets: pd.DataFrame) -> dict:
    """Retorna métricas para um subset do df_bets."""
    placed = df_bets[df_bets["bet_placed"]]
    if placed.empty:
        return {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_rate": None, "roi": None, "total_profit": None}

    wins   = (placed["outcome"] == "WIN").sum()
    losses = (placed["outcome"] == "LOSS").sum()
    pushes = (placed["outcome"] == "PUSH").sum()
    total_profit = placed["profit"].sum()
    n = len(placed)
    roi = round(total_profit / n * 100, 2) if n > 0 else None

    return {
        "bets":         n,
        "wins":         int(wins),
        "losses":       int(losses),
        "pushes":       int(pushes),
        "win_rate":     round(wins / (n - pushes) * 100, 1) if (n - pushes) > 0 else None,
        "roi":          roi,
        "total_profit": round(float(total_profit), 2),
    }


def compute_metrics(df_bets: pd.DataFrame) -> dict:
    """Métricas globais + breakdowns por snap_minute, edge band e liga."""
    global_m = _metrics_for(df_bets)

    # Breakdown por snap_minute
    by_minute = {}
    for minute, grp in df_bets.groupby("snap_minute"):
        by_minute[int(minute)] = _metrics_for(grp)

    # Breakdown por faixa de edge
    by_edge: dict[str, dict] = {}
    for lo, hi in EDGE_BANDS:
        label = f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else f"{lo:.1f}+"
        mask  = (abs(df_bets["edge"]) >= lo) & (abs(df_bets["edge"]) < hi) & df_bets["bet_placed"]
        by_edge[label] = _metrics_for(df_bets[mask])

    # Breakdown por liga (top 10 por volume de apostas)
    by_league: dict[str, dict] = {}
    if "league_name" in df_bets.columns:
        placed = df_bets[df_bets["bet_placed"]]
        top_leagues = placed["league_name"].value_counts().head(10).index
        for lg in top_leagues:
            by_league[str(lg)] = _metrics_for(placed[placed["league_name"] == lg])

    return {
        "global":     global_m,
        "by_minute":  by_minute,
        "by_edge":    by_edge,
        "by_league":  by_league,
    }


# ---------------------------------------------------------------------------
# Exibição e salvamento
# ---------------------------------------------------------------------------

def print_report(metrics: dict, edge_threshold: float, odds_value: float) -> None:
    g = metrics["global"]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT  |  edge≥{edge_threshold}  |  odds={odds_value}")
    print(sep)
    print(f"  Apostas totais : {g['bets']}")
    print(f"  Vitórias       : {g['wins']} | Derrotas: {g['losses']} | Push: {g['pushes']}")
    wr = f"{g['win_rate']:.1f}%" if g["win_rate"] is not None else "N/A"
    roi_s = f"{g['roi']:+.2f}%" if g["roi"] is not None else "N/A"
    pnl = f"{g['total_profit']:+.2f}u" if g["total_profit"] is not None else "N/A"
    print(f"  Win rate       : {wr}")
    print(f"  ROI            : {roi_s}")
    print(f"  P&L total      : {pnl}")

    print(f"\n  {'BREAKDOWN POR MINUTO':}")
    print(f"  {'Min':>6}  {'Apostas':>8}  {'WinRate':>8}  {'ROI':>8}  {'P&L':>8}")
    print("  " + "-" * 50)
    for minute, m in sorted(metrics["by_minute"].items()):
        if m["bets"] == 0:
            continue
        wr_m  = f"{m['win_rate']:.1f}%"  if m["win_rate"]     is not None else "  N/A"
        roi_m = f"{m['roi']:+.2f}%"      if m["roi"]          is not None else "  N/A"
        pnl_m = f"{m['total_profit']:+.2f}u" if m["total_profit"] is not None else "  N/A"
        print(f"  {minute:>6}  {m['bets']:>8}  {wr_m:>8}  {roi_m:>8}  {pnl_m:>8}")

    print(f"\n  {'BREAKDOWN POR FAIXA DE EDGE':}")
    print(f"  {'Edge':>8}  {'Apostas':>8}  {'WinRate':>8}  {'ROI':>8}  {'P&L':>8}")
    print("  " + "-" * 50)
    for band, m in metrics["by_edge"].items():
        if m["bets"] == 0:
            continue
        wr_e  = f"{m['win_rate']:.1f}%"      if m["win_rate"]     is not None else "  N/A"
        roi_e = f"{m['roi']:+.2f}%"          if m["roi"]          is not None else "  N/A"
        pnl_e = f"{m['total_profit']:+.2f}u" if m["total_profit"] is not None else "  N/A"
        print(f"  {band:>8}  {m['bets']:>8}  {wr_e:>8}  {roi_e:>8}  {pnl_e:>8}")

    if metrics["by_league"]:
        print(f"\n  {'BREAKDOWN POR LIGA (top 10)':}")
        print(f"  {'Liga':<30}  {'Apostas':>7}  {'WinRate':>8}  {'ROI':>8}  {'P&L':>8}")
        print("  " + "-" * 68)
        for league, m in sorted(metrics["by_league"].items(),
                                 key=lambda x: -(x[1]["bets"] or 0)):
            if m["bets"] == 0:
                continue
            wr_l  = f"{m['win_rate']:.1f}%"      if m["win_rate"]     is not None else "  N/A"
            roi_l = f"{m['roi']:+.2f}%"          if m["roi"]          is not None else "  N/A"
            pnl_l = f"{m['total_profit']:+.2f}u" if m["total_profit"] is not None else "  N/A"
            print(f"  {league:<30}  {m['bets']:>7}  {wr_l:>8}  {roi_l:>8}  {pnl_l:>8}")
    print(f"\n{sep}\n")


def save_report(df_bets: pd.DataFrame, path: Path) -> None:
    cols = [
        "event_id", "snap_minute", "league_name", "home_team", "away_team",
        "kickoff_dt", "corners_line", "pred", "edge",
        "bet_placed", "bet_direction",
        "target_corners_total", "outcome", "profit",
    ]
    cols = [c for c in cols if c in df_bets.columns]
    df_bets[df_bets["bet_placed"]][cols].to_csv(path, index=False, encoding="utf-8")
    log.info("Relatório salvo em: %s", path)


def plot_profit_curve(df_bets: pd.DataFrame, edge_threshold: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib não instalado. Instale com: pip install matplotlib")
        return

    placed = df_bets[df_bets["bet_placed"]].copy().reset_index(drop=True)
    if placed.empty:
        log.warning("Nenhuma aposta para plotar.")
        return

    placed["cumprofit"] = placed["profit"].cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de profit acumulado
    axes[0].plot(placed.index + 1, placed["cumprofit"], color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Profit Acumulado (edge≥{edge_threshold})")
    axes[0].set_xlabel("Nº de apostas")
    axes[0].set_ylabel("Unidades apostadas")
    axes[0].grid(alpha=0.3)

    # Distribuição do edge
    axes[1].hist(placed["edge"], bins=30, color="coral", edgecolor="white")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Distribuição de Edge (apostas realizadas)")
    axes[1].set_xlabel("Edge (previsão − linha)")
    axes[1].set_ylabel("Frequência")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "backtest_profit_curve.png"
    plt.savefig(out, dpi=120)
    plt.show()
    log.info("Plot salvo em: %s", out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtesting do modelo de escanteios")
    p.add_argument("--edge",     type=float, default=0.5,  help="Edge mínimo para apostar")
    p.add_argument("--odds",     type=float, default=1.85, help="Odds assumida para O/U")
    p.add_argument("--snapshot", type=int,   default=None,
                   help="Filtrar por snap_minute específico (ex: 45)")
    p.add_argument("--plot",     action="store_true",      help="Gerar gráfico de profit")
    p.add_argument("--output",   default=str(REPORT_PATH), help="CSV de saída")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not FEATURES_PATH.exists():
        log.error("features_ml.parquet não encontrado em %s", FEATURES_PATH)
        log.error("Execute primeiro: python betsapi_corners_analysis.py")
        return

    log.info("Carregando features_ml.parquet...")
    df_features = pd.read_parquet(FEATURES_PATH)
    log.info("Linhas: %d | Jogos: %d", len(df_features), df_features["event_id"].nunique())

    model = load_model()

    log.info("Simulando apostas (edge≥%.1f, odds=%.2f)...", args.edge, args.odds)
    df_bets = simulate_bets(df_features, model, args.edge, args.odds, args.snapshot)

    metrics = compute_metrics(df_bets)
    print_report(metrics, args.edge, args.odds)

    save_report(df_bets, Path(args.output))

    if args.plot:
        plot_profit_curve(df_bets, args.edge)


if __name__ == "__main__":
    main()
