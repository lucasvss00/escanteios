# ✓ Correção Implementada: Escanteios no Panorama

## O que foi feito

**Arquivo modificado:** `betsapi_corners_collector.py`

**Função:** `build_panorama_row()` (linhas 402-407)

**Mudança:** Adicionado fallback para agregar escanteios dos snapshots quando a API não fornece

```python
# Fallback: se a API (/event/view) não forneceu stats de escanteios,
# usa o último valor dos snapshots (que vêm do stats_trend)
if corners_h is None and snapshot_rows:
    corners_h = _safe_last(snapshot_rows, "corners_home")
if corners_a is None and snapshot_rows:
    corners_a = _safe_last(snapshot_rows, "corners_away")
```

---

## Como testar

### Opção 1: Teste rápido (verificar compilação)

```bash
cd C:\Users\Lucas\Desktop\FIFA\escanteios

# Ativar ambiente conda
conda activate Betsapi

# Testar compilação
python -c "import py_compile; py_compile.compile('betsapi_corners_collector.py', doraise=True); print('✓ OK')"
```

### Opção 2: Teste completo (validar dados)

```bash
cd C:\Users\Lucas\Desktop\FIFA\escanteios

# Rodar script de teste
python test_fix.py
```

O script fará:
1. ✓ Verificar compilação (sem erros de sintaxe)
2. ✓ Comparar dados de snapshots vs panorama
3. ✓ Validar que escanteios estão preenchidos

### Opção 3: Coleta de teste com a correção

```bash
cd C:\Users\Lucas\Desktop\FIFA\escanteios

# Coleta de 1 dia, máximo 5 jogos
python betsapi_corners_collector.py \
  --mode historico \
  --start 20260325 \
  --end 20260325 \
  --max-games 5 \
  --token SEU_TOKEN
```

Depois, verifique em Python:

```python
import pandas as pd

# Carregar dados
pano = pd.read_csv("dados_escanteios/panorama_jogos.csv")

# Verificar que corners_total NÃO está vazio
print(pano[["event_id", "corners_home_total", "corners_away_total", "corners_total"]].head())

# Quantos jogos têm corners preenchido?
filled = (pano["corners_total"] > 0).sum()
print(f"\nJogos com corners > 0: {filled}/{len(pano)}")
```

---

## Impacto da correção

| Aspecto | Antes | Depois |
|---------|-------|--------|
| `corners_home_total` | Vazio (None) | Preenchido com valor real |
| `corners_away_total` | Vazio (None) | Preenchido com valor real |
| `corners_total` | Sempre 0 | Valor correto (h + a) |
| `corners_home_ht` | Calculado errado | Correto (usa corners_total real) |
| `corners_home_2h` | Calculado errado | Correto (usa corners_total real) |

---

## Fluxo de Dados com a Correção

```
API /v1/event/stats_trend
    ↓ (parse_stats_trend)
Snapshots: 1 linha por minuto com corners_home/away
    ↓
Salvo em: snapshots_por_minuto.parquet ✓ (tinha dados)
    ↓
    + API /v1/event/view (stats)
    ↓ (build_panorama_row)
Panorama: 1 linha por jogo com corners_home_total/away_total
    ↓ NOVO: Se API não deu, usa _safe_last dos snapshots
    ↓
Salvo em: panorama_jogos.parquet ✓ (agora com dados preenchidos)
```

---

## Próximos passos

Após validar a correção:

1. **Coleta em larga escala** — rodar para vários dias com confiança de dados completos
2. **Feature engineering** — usar `panorama_jogos.parquet` com corners totais confiáveis
3. **Treino de ML** — dados prontos para modelo XGBoost

---

## Suporte

Se algum teste falhar, verifique:
- ✓ Python 3.11+ instalado
- ✓ Dependências instaladas: `pandas`, `pyarrow`
- ✓ Arquivo `betsapi_corners_collector.py` foi modificado (linha 402-407 com fallback)
