#!/usr/bin/env python3
"""
Script de teste para validar a correção de escanteios no panorama.
Execute com: python test_fix.py
"""

import sys
import pandas as pd
from pathlib import Path

def test_compilation():
    """Testa se o módulo compila sem erros de sintaxe."""
    print("=" * 70)
    print("1. Testando compilação do betsapi_corners_collector.py...")
    print("=" * 70)
    try:
        import py_compile
        py_compile.compile('betsapi_corners_collector.py', doraise=True)
        print("✓ Compilação OK — nenhum erro de sintaxe\n")
        return True
    except Exception as e:
        print(f"✗ Erro de compilação: {e}\n")
        return False


def test_data_integrity():
    """Verifica se os dados coletados têm escanteios preenchidos."""
    print("=" * 70)
    print("2. Testando integridade dos dados coletados...")
    print("=" * 70)

    data_dir = Path("dados_escanteios")
    if not data_dir.exists():
        print("✗ Diretório 'dados_escanteios' não encontrado")
        return False

    snap_file = data_dir / "snapshots_por_minuto.csv"
    pano_file = data_dir / "panorama_jogos.csv"

    if not snap_file.exists() or not pano_file.exists():
        print(f"✗ Arquivos de dados não encontrados")
        return False

    # Ler os dados
    snap = pd.read_csv(snap_file)
    pano = pd.read_csv(pano_file)

    print(f"Snapshots coletados: {len(snap)} linhas")
    print(f"Jogos únicos: {len(pano)} jogos\n")

    # Verificação 1: Snapshots têm dados de escanteios?
    snap_corners_filled = snap["corners_home"].notna().sum()
    print(f"Snapshots com corners_home preenchido: {snap_corners_filled}/{len(snap)}")

    # Verificação 2: Panorama tem corners_total preenchido?
    pano_corners_filled = pano["corners_total"].notna().sum()
    pano_corners_zero = (pano["corners_total"] == 0).sum()
    pano_corners_empty = pano["corners_total"].isna().sum()

    print(f"Panorama com corners_total preenchido: {pano_corners_filled}/{len(pano)}")
    print(f"Panorama com corners_total = 0: {pano_corners_zero}")
    print(f"Panorama com corners_total vazio: {pano_corners_empty}\n")

    # Verificação 3: Exemplo específico
    print("Exemplos de consistência (snap → panorama):")
    print("-" * 70)

    success_count = 0
    for event_id in snap["event_id"].unique()[:3]:
        snap_subset = snap[snap["event_id"] == event_id]
        pano_subset = pano[pano["event_id"] == event_id]

        if snap_subset.empty or pano_subset.empty:
            continue

        # Último snapshot de cada time
        snap_h = snap_subset["corners_home"].iloc[-1]
        snap_a = snap_subset["corners_away"].iloc[-1]
        snap_total = (snap_h or 0) + (snap_a or 0) if snap_h is not None and snap_a is not None else None

        # Panorama
        pano_h = pano_subset["corners_home_total"].iloc[0]
        pano_a = pano_subset["corners_away_total"].iloc[0]
        pano_total = pano_subset["corners_total"].iloc[0]

        match = (snap_h == pano_h) and (snap_a == pano_a)
        status = "✓" if match else "✗"

        print(f"{status} {event_id}:")
        print(f"   Snapshots: home={snap_h} away={snap_a} total={snap_total}")
        print(f"   Panorama:  home={pano_h} away={pano_a} total={pano_total}")

        if match:
            success_count += 1

    print("-" * 70)
    print(f"Consistência: {success_count}/3 jogos com valores corretos\n")

    return success_count == 3


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("TESTE DA CORREÇÃO DE ESCANTEIOS NO PANORAMA")
    print("=" * 70 + "\n")

    # Teste 1: Compilação
    compile_ok = test_compilation()

    # Teste 2: Integridade de dados
    data_ok = False
    if compile_ok:
        try:
            data_ok = test_data_integrity()
        except Exception as e:
            print(f"Erro ao testar dados: {e}\n")

    # Resumo
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"Compilação: {'✓ OK' if compile_ok else '✗ FALHOU'}")
    print(f"Dados:      {'✓ OK' if data_ok else '✗ FALHOU'}\n")

    if compile_ok and data_ok:
        print("✓ Todos os testes passaram! A correção está funcionando.\n")
        return 0
    else:
        print("✗ Alguns testes falharam. Veja os detalhes acima.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
