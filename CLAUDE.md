# Projeto: Modelo Preditivo de Escanteios — BetsAPI

## Repositório GitHub
- **URL**: https://github.com/lucasvss00/escanteios
- **Branch principal**: `main`
- **Auto-push**: cada arquivo editado/criado pelo Claude Code dispara automaticamente `git add -A && git commit && git push origin main` via hook `PostToolUse` configurado em `.claude/settings.json`

## Descrição
Coleta, processamento e modelagem de dados de escanteios de futebol via BetsAPI (b365api.com).
O objetivo é treinar um modelo de ML capaz de prever o total de escanteios de uma partida
tanto no pré-jogo quanto ao vivo (em tempo real, por minuto de snapshot).

---

## Estrutura de arquivos

```
escanteios/
├── betsapi_corners_collector.py   # Coleta histórica e ao vivo via BetsAPI
├── betsapi_corners_analysis.py    # Feature engineering + treino XGBoost
├── dados_escanteios/              # Gerado automaticamente pelo coletor
│   ├── snapshots_por_minuto.parquet
│   ├── panorama_jogos.parquet
│   ├── features_ml.parquet
│   └── modelo_corners_xgb.joblib
└── CLAUDE.md
```

---

## Stack / Dependências

```bash
pip install requests pandas pyarrow xgboost scikit-learn joblib tqdm
```

- Python 3.11+
- `requests` — chamadas à BetsAPI
- `pandas` + `pyarrow` — armazenamento em Parquet
- `xgboost` + `scikit-learn` — modelo preditivo
- `tqdm` — barra de progresso na coleta histórica

---

## Como rodar

### Coleta histórica (jogos finalizados)
```bash
# Últimos 30 dias
python betsapi_corners_collector.py --mode historico --days 30 --token SEU_TOKEN

# Range específico
python betsapi_corners_collector.py --mode historico --start 20240101 --end 20240131 --token SEU_TOKEN

# Filtrar por liga específica
python betsapi_corners_collector.py --mode historico --days 30 --league-id 1 --token SEU_TOKEN
```

### Coleta ao vivo (loop contínuo)
```bash
python betsapi_corners_collector.py --mode live --interval 60 --token SEU_TOKEN
```

### Análise e treino do modelo
```bash
python betsapi_corners_analysis.py
```

---

## Endpoints BetsAPI utilizados

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/v3/events/ended` | Jogos finalizados por dia |
| GET | `/v3/events/inplay` | Jogos ao vivo |
| GET | `/v1/event/view` | Detalhes e placar final do jogo |
| GET | `/v1/event/stats_trend` | Série temporal de stats por minuto |
| GET | `/v3/bet365/prematch` | Odds pré-jogo Bet365 (escanteios) |

---

## Dados coletados

### `snapshots_por_minuto.parquet`
Série temporal — 1 linha por jogo × minuto.

| Campo | Descrição |
|-------|-----------|
| `corners_home/away` | Escanteios acumulados |
| `attacks_home/away` | Ataques |
| `dangerous_attacks_home/away` | Ataques perigosos |
| `shots_on/off_target_home/away` | Chutes a gol / fora |
| `possession_home/away` | Posse de bola (%) |
| `yellow/red_cards_home/away` | Cartões |
| `fouls_home/away` | Faltas |
| `saves_home/away` | Defesas do goleiro |
| `offsides_home/away` | Impedimentos |
| `goal_kicks_home/away` | Tiros de meta |

### `panorama_jogos.parquet`
Resumo final — 1 linha por jogo.
Inclui totais finais de todas as stats acima, placar final, placar do intervalo,
escanteios por tempo (1º e 2º), odds de escanteios pré-jogo (linha Over/Under e Asian Corners).

---

## Features de ML (`features_ml.parquet`)

Snapshots nos minutos: **15, 30, 45, 60, 75**.

Para cada snapshot:
- Stats acumuladas até aquele minuto
- Ritmo de escanteios por minuto
- Diferenças home − away (pressão assimétrica)
- Variação de escanteios nos últimos 15 minutos
- Placar ao vivo
- Posse média

**Targets:**
- `target_corners_total` — total de escanteios ao final (regressão)
- `target_corners_remaining` — escanteios que ainda faltam (regressão)
- `target_more_corners` — haverá pelo menos mais 1 escanteio? (binário)

---

## Convenções de código

- Sempre usar **type hints** em funções novas
- Token da API **sempre via argumento CLI `--token`** — nunca hardcoded
- Dados persistidos em **Parquet** como formato principal; CSV é opcional (`--no-csv` para desativar)
- `REQUEST_DELAY = 0.35s` entre chamadas para respeitar o rate limit da API
- Logging via `logging` padrão (não usar `print` em produção)
- Funções de parse retornam `None` em caso de dado ausente/inválido (nunca lançar exceção)

---

## Observações importantes

- O `stats_trend` só está disponível para jogos a partir de **2017-06-10**
- O `FIELD_MAP` mapeia os índices 0–11 do array retornado pela API; índices além de 11 são ignorados
- O placar do intervalo é extraído via `event/view → results[0].scores["1"]`
- Escanteios por tempo são calculados: 1º tempo = minuto ≤ 45; 2º tempo = total − 1º tempo
- Odds de escanteios podem ser `None` se a Bet365 não cobriu o mercado para aquele jogo
