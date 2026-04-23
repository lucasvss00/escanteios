"""
run_diagnostics.py
==================
Orquestrador de todos os scripts de diagnóstico do modelo de escanteios.

MODOS:
    python run_diagnostics.py                # modo rápido (padrão, ~5-10 min)
    python run_diagnostics.py --full         # tudo inclusive walk-forward (~60-90 min)
    python run_diagnostics.py --step 1 3     # só passos 1 e 3
    python run_diagnostics.py --list         # lista os passos e sai

PASSOS DISPONÍVEIS:
    1  diagnose_dead_features   — features com SHAP=0 (rápido, ~1 min)
    2  baseline_comparison      — MAE vs baselines triviais (~2 min)
    3  audit_team_encoding      — auditoria do target encoding (~3 min)
    4  calibration_diagnosis    — PIT + ECE do NGBoost (~3 min)
    5  recalibrate_ngboost      — calibração isotônica NGBoost (~5 min)
    6  validate_min45           — análise profunda minuto 45 (~15 min)
    ── EXPERIMENTOS LENTOS (só com --full) ──
    7  diagnose_dead_features   [--sanity]         (~10 min extra)
    8  audit_team_encoding      [--experiment]     (~30 min extra)
    9  recalibrate_ngboost      [--walk-forward]   (~20 min extra)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# DEFINIÇÃO DOS PASSOS
# ---------------------------------------------------------------------------
PYTHON = sys.executable   # usa o mesmo interpretador Python do ambiente atual

# Cada passo: (id, label, comando_rápido, comando_full, estimativa_rápida_min)
STEPS = [
    (1, "diagnose_dead_features",
     ["diagnose_dead_features.py"],
     ["diagnose_dead_features.py", "--sanity"],
     1),

    (2, "baseline_comparison",
     ["baseline_comparison.py"],
     ["baseline_comparison.py"],
     2),

    (3, "audit_team_encoding  (auditoria)",
     ["audit_team_encoding.py"],
     ["audit_team_encoding.py", "--experiment", "--n-estimators", "200"],
     3),

    (4, "calibration_diagnosis (PIT + ECE)",
     ["calibration_diagnosis.py"],
     ["calibration_diagnosis.py"],
     3),

    (5, "recalibrate_ngboost  (calibração isotônica)",
     ["recalibrate_ngboost.py"],
     ["recalibrate_ngboost.py", "--walk-forward", "--n-estimators", "150"],
     5),

    (6, "validate_min45       (análise profunda)",
     ["validate_min45.py"],
     ["validate_min45.py"],
     15),
]

RESULTS_DIR = Path("dados_escanteios")
LOG_DIR     = Path("dados_escanteios/logs_diagnostico")

# CSVs produzidos por cada passo (para o resumo final)
STEP_OUTPUTS = {
    1: ["dead_features_report.csv"],
    2: ["baseline_comparison.csv", "baseline_bucket_analysis.csv",
        "baseline_betting_comparison.csv"],
    3: ["team_encoding_audit.csv"],
    4: ["ngboost_calibration_report.csv"],
    5: ["recalibration_report.csv"],
    6: ["min45_decision.csv"],
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds/60:.1f}min"


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"


def _print_header(step_id: int, label: str, total: int) -> None:
    print(f"\n{'═'*70}")
    print(f"  PASSO {step_id}/{total}  —  {label}")
    print(f"{'═'*70}")


def _print_separator() -> None:
    print(f"\n{'─'*70}")


def _run_step(
    step_id: int,
    label: str,
    cmd_args: list[str],
    log_path: Path,
    dry_run: bool = False,
) -> tuple[bool, float]:
    """
    Executa um passo. Faz stream do stdout em tempo real E salva no log.
    Retorna (sucesso, duração_segundos).
    """
    full_cmd = [PYTHON] + cmd_args
    print(f"  $ {' '.join(full_cmd)}")
    print(f"  Log → {log_path}\n")

    if dry_run:
        print("  [dry-run] Passo não executado.")
        return True, 0.0

    t0 = time.time()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as flog:
            flog.write(f"# {datetime.now().isoformat()}  cmd: {' '.join(full_cmd)}\n\n")
            import os as _os
            _env = _os.environ.copy()
            _env["PYTHONIOENCODING"] = "utf-8"
            _env["PYTHONUTF8"] = "1"
            proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_env,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                flog.write(line)
            proc.wait()
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        status = "✅ OK" if ok else f"❌ ERRO (returncode={proc.returncode})"
        print(f"\n  {status}  ({_fmt_time(elapsed)})")
        return ok, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ Falha ao executar: {e}")
        return False, elapsed


def _summarise_results() -> None:
    """Lê os CSVs de output e imprime um resumo executivo."""
    import pandas as pd

    print(f"\n{'═'*70}")
    print("  RESUMO EXECUTIVO — outputs gerados")
    print(f"{'═'*70}")

    # 1. Features mortas
    p = RESULTS_DIR / "dead_features_report.csv"
    if p.exists():
        df = pd.read_csv(p)
        counts = df["diagnostic"].value_counts()
        print(f"\n  [1] Dead features:")
        for diag, n in counts.items():
            print(f"      {diag:<35} {n:>4}")

    # 2. Baseline comparison
    p = RESULTS_DIR / "baseline_comparison.csv"
    if p.exists():
        df = pd.read_csv(p)
        print(f"\n  [2] Ganho XGBoost vs Baseline 1 (extrapolação linear):")
        if "snap_min" in df.columns and "ganho_vs_B1_pct" in df.columns:
            for _, row in df.iterrows():
                g = row.get("ganho_vs_B1_pct", float("nan"))
                m = int(row["snap_min"])
                v = ("✅" if g > 15 else "⚠" if g > 5 else "❌") if not pd.isna(g) else "?"
                print(f"      min{m}: ganho={g:+.1f}%  {v}")

    # 3. Encoding audit
    p = RESULTS_DIR / "team_encoding_audit.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "corr_drift" in df.columns:
            worst = df.nlargest(3, "corr_drift")[["snap_min", "column", "corr_drift",
                                                   "corr_train", "corr_test"]]
            print(f"\n  [3] Target encoding — top 3 drifts (corr_train vs corr_test):")
            for _, row in worst.iterrows():
                print(f"      min{int(row['snap_min'])} {row['column']:<12} "
                      f"drift={row['corr_drift']:.4f}  "
                      f"train={row['corr_train']:.4f} test={row['corr_test']:.4f}")

    # 4. Calibração NGBoost
    p = RESULTS_DIR / "ngboost_calibration_report.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "ks_pval" in df.columns:
            print(f"\n  [4] Calibração NGBoost (KS p-value — p<0.05 = descalibrado):")
            for _, row in df.iterrows():
                pval = row.get("ks_pval", float("nan"))
                ece  = row.get("ece_lognorm", row.get("ece_lognorm_raw", float("nan")))
                m    = int(row["snap_min"])
                ok   = "✅" if pval >= 0.05 else "❌"
                print(f"      min{m}: KS p={pval:.4f} {ok}  ECE={ece:.5f}")

    # 5. Recalibração
    p = RESULTS_DIR / "recalibration_report.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "ece_lognorm_raw" in df.columns:
            print(f"\n  [5] Recalibração isotônica NGBoost:")
            for _, row in df.iterrows():
                m    = int(row["snap_min"])
                raw  = row.get("ece_lognorm_raw", float("nan"))
                rcal = row.get("ece_lognorm_recal", float("nan"))
                imp  = (raw - rcal) / max(raw, 1e-9) * 100
                adpt = str(row.get("adopted", "?"))
                print(f"      min{m}: ECE {raw:.5f} → {rcal:.5f} ({imp:+.0f}%)  "
                      f"adotado={adpt}")

    # 6. Decisão min 45
    p = RESULTS_DIR / "min45_decision.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "decision" in df.columns and len(df) > 0:
            row = df.iloc[0]
            dec = row["decision"]
            ic  = row.get("ic90_lo_best", float("nan"))
            n   = row.get("n_bets_at_best", 0)
            dr  = row.get("last_third_roi", float("nan"))
            icon = "✅" if dec == "MANTER" else "❌" if dec == "DESCARTAR" else "🔶"
            print(f"\n  [6] Minuto 45: {icon} {dec}")
            print(f"      IC90% lo máximo: {ic:+.2%}  (n={int(n)} apostas)")
            print(f"      ROI último terço: {float(dr):+.2%}")

    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Orquestrador dos scripts de diagnóstico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--full",     action="store_true",
                        help="Inclui experimentos walk-forward lentos (~60-90 min)")
    parser.add_argument("--step",     type=int, nargs="+",
                        help="Executa só os passos especificados (ex: --step 1 3 5)")
    parser.add_argument("--list",     action="store_true",
                        help="Lista passos e estimativas de tempo sem executar")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Imprime os comandos sem executar")
    parser.add_argument("--no-summary", action="store_true",
                        help="Pula o resumo executivo final")
    parser.add_argument("--dir",      default=".",
                        help="Diretório raiz do projeto (padrão: .)")
    args = parser.parse_args()

    # Muda para o diretório do projeto
    import os
    os.chdir(Path(args.dir).resolve())

    # ── Lista e sai ──
    if args.list:
        total_quick = sum(s[4] for s in STEPS)
        total_full  = total_quick + 10 + 30 + 20  # sanity + experiment + wf
        print(f"\n  Passos disponíveis  (modo rápido ≈{total_quick} min | "
              f"modo --full ≈{total_full} min)\n")
        for sid, label, _, _, est in STEPS:
            print(f"  [{sid}] {label:<45} ~{est} min")
        print()
        print("  Experimentos extras (só com --full):")
        print("  [7] diagnose_dead_features --sanity          ~10 min")
        print("  [8] audit_team_encoding    --experiment       ~30 min")
        print("  [9] recalibrate_ngboost    --walk-forward     ~20 min")
        return

    # ── Seleciona quais passos rodar ──
    if args.step:
        selected = [s for s in STEPS if s[0] in args.step]
        if not selected:
            print(f"[ERRO] Nenhum passo válido em {args.step}. Use --list para ver os IDs.")
            sys.exit(1)
    else:
        selected = STEPS

    total  = len(selected)
    mode   = "FULL" if args.full else "RÁPIDO"
    est_total = sum(s[4] for s in selected) + (60 if args.full else 0)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║         DIAGNÓSTICO DO MODELO DE ESCANTEIOS                         ║
║  Modo: {mode:<10}  Passos: {total}   Estimativa: ~{est_total} min           ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t_global = time.time()

    for i, (sid, label, cmd_quick, cmd_full, est_min) in enumerate(selected, 1):
        cmd = cmd_full if args.full else cmd_quick
        log_path = LOG_DIR / f"step{sid:02d}_{cmd[0].replace('.py','')}__{ts}.log"
        _print_header(sid, label, total)
        print(f"  {_bar(i - 1, total)}  restante ≈{_fmt_time((total-i+1)*est_min*60)}\n")

        ok, elapsed = _run_step(sid, label, cmd, log_path, dry_run=args.dry_run)
        results.append({"step": sid, "label": label, "ok": ok,
                        "elapsed": elapsed, "log": str(log_path)})
        _print_separator()

    # ── Sumário de execução ──
    total_elapsed = time.time() - t_global
    n_ok  = sum(1 for r in results if r["ok"])
    n_err = len(results) - n_ok

    print(f"\n{'═'*70}")
    print(f"  EXECUÇÃO CONCLUÍDA  ({_fmt_time(total_elapsed)})")
    print(f"  Passos: {n_ok} ✅  {n_err} ❌")
    print(f"{'═'*70}")

    for r in results:
        icon = "✅" if r["ok"] else "❌"
        print(f"  {icon} [{r['step']}] {r['label']:<45} {_fmt_time(r['elapsed'])}")
        if not r["ok"]:
            print(f"     → log: {r['log']}")

    # ── Resumo executivo dos outputs ──
    if not args.no_summary and not args.dry_run and n_ok > 0:
        try:
            import pandas as pd
            _summarise_results()
        except ImportError:
            print("\n  (pandas não disponível para resumo automático)")

    if n_err > 0:
        print(f"\n  ⚠ {n_err} passo(s) falharam. Verifique os logs em: {LOG_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
