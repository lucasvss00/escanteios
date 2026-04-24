"""
_roi_utils.py
=============
Utilitários de ROI compartilhados entre os scripts de diagnóstico.
Replica a lógica exata de betsapi_corners_analysis.py:

  - _dline_vec  : linha dinâmica arredondada para 0.5 mais próximo
  - profit_vec  : lucro/prejuízo usando odds reais por jogo quando disponíveis
  - select_thresh: threshold robusto no CAL (Over E Under); retorna (roi_te, n, thresh, side)
"""
from __future__ import annotations

import numpy as np

# ── Constantes idênticas ao pipeline principal ──────────────────────────────
ODDS_OVER        = 1.83
ODDS_UNDER       = 1.83
BREAKEVEN        = 1.0 / ODDS_OVER    # ≈ 0.546
MIN_EDGE         = 0.02
_THRESH_MAX      = 0.72
_MIN_CAL_BET     = 50     # fallback: mínimo de apostas no cal
_MIN_CAL_ROI     = 0.05   # fallback: ROI mínimo de 5% no cal
_ROBUST_MIN_ROI  = 0.15   # critério robusto: ROI alvo
_ROBUST_MIN_BETS = 200    # critério robusto: apostas mínimas


def dline_vec(csf_arr, la_arr_or_list, rem: int) -> np.ndarray:
    """
    Linha dinâmica por liga, arredondada para o 0.5 mais próximo.
    Idêntico a _dline_vec em betsapi_corners_analysis.py linha 2635.
    """
    lines = []
    la_arr = la_arr_or_list if la_arr_or_list is not None else [None] * len(csf_arr)
    for csf, la in zip(csf_arr, la_arr):
        try:
            rate = (float(la) / 90.0
                    if (la is not None and np.isfinite(float(la)) and float(la) > 0)
                    else 0.1)
        except (TypeError, ValueError):
            rate = 0.1
        lines.append(np.round((float(csf) + rem * rate) * 2) / 2)
    return np.array(lines, dtype=float)


def profit_vec(
    over_actual: np.ndarray,
    mask: np.ndarray,
    odds_arr,                   # array-like ou None → usa default_odds
    default_odds: float,
    is_over: bool = True,
) -> tuple[float, int]:
    """
    Calcula (ROI, n_apostas) usando odds reais por jogo quando disponíveis.
    Idêntico a _profit_vec em betsapi_corners_analysis.py linha 2955.
    """
    idxs = np.where(mask)[0]
    n = len(idxs)
    if n == 0:
        return 0.0, 0
    profit = 0.0
    for i in idxs:
        try:
            o = (float(odds_arr[i])
                 if (odds_arr is not None
                     and not np.isnan(float(odds_arr[i]))
                     and float(odds_arr[i]) > 1.0)
                 else default_odds)
        except (TypeError, ValueError):
            o = default_odds
        won = int(over_actual[i]) if is_over else int(1 - over_actual[i])
        profit += (o - 1) * won - (1 - won)
    return float(profit / n), n


def select_thresh(
    p_over_cal: np.ndarray,
    over_actual_cal: np.ndarray,
    p_over_te: np.ndarray,
    over_actual_te: np.ndarray,
    odds_over_arr=None,    # odds reais Over no TEST
    odds_under_arr=None,   # odds reais Under no TEST
) -> tuple[float, int, float, str]:
    """
    Seleciona threshold no CAL com critério robusto (ROI≥15% + n≥200).
    Avalia Over E Under — escolhe o melhor lado.
    Retorna (roi_te, n_bets, threshold, side).

    Idêntico à lógica de seleção de threshold em
    betsapi_corners_analysis.py linhas 2885–2950.
    """
    p_under_cal      = 1.0 - p_over_cal
    under_actual_cal = 1 - over_actual_cal

    # ── Sweep Over no CAL ────────────────────────────────────────────────────
    best_over_thresh, best_over_roi = BREAKEVEN + MIN_EDGE, -999.0
    _robust_over_found = False
    for _thr in np.arange(BREAKEVEN + MIN_EDGE, _THRESH_MAX + 0.001, 0.01):
        mc = p_over_cal >= _thr
        nc = int(mc.sum())
        if nc < _MIN_CAL_BET:
            continue
        wc = over_actual_cal[mc].sum()
        rc = (wc * (ODDS_OVER - 1) - (nc - wc)) / nc
        if not _robust_over_found and nc >= _ROBUST_MIN_BETS and rc >= _ROBUST_MIN_ROI:
            best_over_roi, best_over_thresh = rc, _thr
            _robust_over_found = True
        if not _robust_over_found and rc > best_over_roi:
            best_over_roi, best_over_thresh = rc, _thr

    # ── Sweep Under no CAL ───────────────────────────────────────────────────
    best_under_thresh, best_under_roi = BREAKEVEN + MIN_EDGE, -999.0
    _robust_under_found = False
    for _thr in np.arange(BREAKEVEN + MIN_EDGE, _THRESH_MAX + 0.001, 0.01):
        mc = p_under_cal >= _thr
        nc = int(mc.sum())
        if nc < _MIN_CAL_BET:
            continue
        wc = under_actual_cal[mc].sum()
        rc = (wc * (ODDS_UNDER - 1) - (nc - wc)) / nc
        if not _robust_under_found and nc >= _ROBUST_MIN_BETS and rc >= _ROBUST_MIN_ROI:
            best_under_roi, best_under_thresh = rc, _thr
            _robust_under_found = True
        if not _robust_under_found and rc > best_under_roi:
            best_under_roi, best_under_thresh = rc, _thr

    # ── Escolhe lado ─────────────────────────────────────────────────────────
    _over_ok  = best_over_roi  >= _MIN_CAL_ROI
    _under_ok = best_under_roi >= _MIN_CAL_ROI

    if _over_ok and best_over_roi >= best_under_roi:
        side, thresh = "Over",  best_over_thresh
    elif _under_ok:
        side, thresh = "Under", best_under_thresh
    elif _over_ok:
        side, thresh = "Over",  best_over_thresh
    else:
        side, thresh = "Over",  BREAKEVEN + MIN_EDGE   # fallback conservador

    # ── Avalia no TEST com odds reais ────────────────────────────────────────
    if side == "Over":
        mask_te = p_over_te >= thresh
        roi_te, n_bets = profit_vec(over_actual_te, mask_te,
                                    odds_over_arr, ODDS_OVER, True)
    else:
        p_under_te = 1.0 - p_over_te
        mask_te    = p_under_te >= thresh
        roi_te, n_bets = profit_vec(over_actual_te, mask_te,
                                    odds_under_arr, ODDS_UNDER, False)

    return roi_te, n_bets, thresh, side
