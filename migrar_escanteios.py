#!/usr/bin/env python3
"""
Migração: Preenche corners_home_total / corners_away_total no panorama
usando os dados de snapshots já coletados — sem chamar a API.

Como funciona:
  1. Lê panorama_jogos.parquet e encontra jogos com corners_total vazio/zero
  2. Para cada jogo, pega o último valor de corners_home/away nos snapshots
  3. Recalcula corners_ht, corners_2h e variáveis derivadas
  4. Sobrescreve o parquet e o CSV com os valores corretos

Uso:
    python migrar_escanteios.py
    python migrar_escanteios.py --dry-run   # só mostra o que seria feito
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path("dados_escanteios")


def safe_last(df_snap: pd.DataFrame, col: str) -> float | None:
    """Último valor não-nulo de uma coluna nos snapshots de um jogo."""
    vals = df_snap[col].dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


def safe_last_filtered(df_snap: pd.DataFrame, col: str, max_minute: int) -> float | None:
    """Último valor não-nulo de uma coluna até max_minute."""
    filtered = df_snap[df_snap["minute"] <= max_minute]
    vals = filtered[col].dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


def recalcular_escanteios(row: pd.Series, df_snap_jogo: pd.DataFrame) -> dict:
    """
    Para um jogo do panorama, recalcula os campos de escanteios
    a partir dos snapshots desse jogo.
    """
    corners_h = safe_last(df_snap_jogo, "corners_home")
    corners_a = safe_last(df_snap_jogo, "corners_away")

    corners_h_ht = safe_last_filtered(df_snap_jogo, "corners_home", 45)
    corners_a_ht = safe_last_filtered(df_snap_jogo, "corners_away", 45)

    c_h = corners_h or 0
    c_a = corners_a or 0
    c_h_ht = corners_h_ht or 0
    c_a_ht = corners_a_ht or 0

    return {
        "corners_home_total":  corners_h,
        "corners_away_total":  corners_a,
        "corners_total":       c_h + c_a,
        "corners_home_ht":     corners_h_ht,
        "corners_away_ht":     corners_a_ht,
        "corners_ht_total":    c_h_ht + c_a_ht,
        "corners_home_2h":     c_h - c_h_ht,
        "corners_away_2h":     c_a - c_a_ht,
        "corners_2h_total":    (c_h + c_a) - (c_h_ht + c_a_ht),
    }


def main():
    parser = argparse.ArgumentParser(description="Migra corners no panorama usando snapshots existentes")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o que seria feito, sem salvar")
    args = parser.parse_args()

    pano_path = DATA_DIR / "panorama_jogos.parquet"
    snap_path = DATA_DIR / "snapshots_por_minuto.parquet"

    # Verifica se os arquivos existem
    if not pano_path.exists():
        print(f"✗ Arquivo não encontrado: {pano_path}")
        sys.exit(1)
    if not snap_path.exists():
        print(f"✗ Arquivo não encontrado: {snap_path}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print("  Migração: Escanteios no Panorama")
    print(f"{'═'*60}")

    # Carrega os dados
    print("\n📂 Carregando dados...")
    df_pano = pd.read_parquet(pano_path)
    df_snap = pd.read_parquet(snap_path)

    df_pano["event_id"] = df_pano["event_id"].astype(str)
    df_snap["event_id"] = df_snap["event_id"].astype(str)

    print(f"   Panorama:  {len(df_pano):,} jogos")
    print(f"   Snapshots: {len(df_snap):,} linhas")

    # Identifica jogos com corners_total ausente ou zero
    if "corners_total" not in df_pano.columns:
        df_pano["corners_total"] = None

    mask_precisa_fix = (
        df_pano["corners_total"].isna() |
        (df_pano["corners_total"] == 0)
    )

    # Mas jogos com corners_total = 0 PODEM ser legítimos (raros mas existem)
    # Filtra apenas jogos que têm snapshots disponíveis E corners nos snapshots
    jogos_problema = df_pano[mask_precisa_fix]["event_id"].tolist()
    snap_ids_com_corners = df_snap.groupby("event_id").apply(
        lambda g: (g["corners_home"].dropna() > 0).any() or (g["corners_away"].dropna() > 0).any()
    )
    snap_ids_com_corners = snap_ids_com_corners[snap_ids_com_corners].index.tolist()

    jogos_a_corrigir = [eid for eid in jogos_problema if eid in snap_ids_com_corners]

    print(f"\n🔎 Diagnóstico:")
    print(f"   Jogos com corners_total = 0 ou vazio: {len(jogos_problema):,}")
    print(f"   Desses, com snapshots com dados de escanteios: {len(jogos_a_corrigir):,}")

    if not jogos_a_corrigir:
        print("\n✓ Nenhum jogo precisa de correção! Tudo já está correto.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Mostrando primeiros 10 jogos que seriam corrigidos:")
        print("-" * 60)
        for eid in jogos_a_corrigir[:10]:
            df_snaps_jogo = df_snap[df_snap["event_id"] == eid].sort_values("minute")
            novos_vals = recalcular_escanteios(None, df_snaps_jogo)
            row = df_pano[df_pano["event_id"] == eid].iloc[0]
            print(f"   {eid}: {row.get('home_team', '?')} vs {row.get('away_team', '?')}")
            print(f"     Antes:  corners_total={row.get('corners_total')} (home={row.get('corners_home_total')} away={row.get('corners_away_total')})")
            print(f"     Depois: corners_total={novos_vals['corners_total']:.0f} (home={novos_vals['corners_home_total']} away={novos_vals['corners_away_total']})")
        print(f"\n[DRY RUN] Execute sem --dry-run para salvar as alterações.")
        return

    # Aplica a correção
    print(f"\n🔧 Corrigindo {len(jogos_a_corrigir):,} jogos...")

    updated = 0
    for eid in jogos_a_corrigir:
        idx = df_pano[df_pano["event_id"] == eid].index
        if idx.empty:
            continue

        df_snap_jogo = df_snap[df_snap["event_id"] == eid].sort_values("minute")
        novos_vals = recalcular_escanteios(None, df_snap_jogo)

        for col, val in novos_vals.items():
            if col not in df_pano.columns:
                df_pano[col] = None
            df_pano.loc[idx, col] = val

        updated += 1

    print(f"   ✓ {updated:,} jogos corrigidos")

    if args.dry_run:
        print("\n[DRY RUN] Nenhum arquivo foi salvo.")
        return

    # Salva os arquivos
    print("\n💾 Salvando arquivos...")
    df_pano.to_parquet(pano_path, index=False)
    print(f"   ✓ {pano_path}")

    csv_path = DATA_DIR / "panorama_jogos.csv"
    df_pano.to_csv(csv_path, index=False)
    print(f"   ✓ {csv_path}")

    # Verifica resultado
    still_zero = (
        df_pano["corners_total"].isna() |
        (df_pano["corners_total"] == 0)
    ).sum()

    print(f"\n{'═'*60}")
    print(f"  Resultado:")
    print(f"    Jogos corrigidos:    {updated:,}")
    print(f"    Ainda sem dados:     {still_zero:,}")
    print(f"      (esses não têm snapshots com escanteios — normal para jogos esparsos)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
