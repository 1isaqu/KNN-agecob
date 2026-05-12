# Ritmo do Dia — Documentação de Modelos ML
**Projeto:** agecob-lens · **Empresa:** Agecob  
**Gerado em:** 2026-05-12 · **Período de dados:** 2025-11-07 → 2026-05-08  
**Última atualização:** retrain em sklearn 1.8.0 (Python 3.14) com features expandidas e k=20

---

## 1. Contexto de Negócio

O endpoint `GET /dashboard/ritmo-dia/{db}` exibe, hora a hora, se a operação de cobrança está no ritmo esperado. A tríade **esperado / real / delta** é o produto; os modelos ML melhoram a qualidade do `esperado` sem alterar a interface.

**Sistema fonte:** COBweb (SQL Server)  
**Bancos:** `COBwebRCBCONSUMER` · `COBwebRCBAUTOS`  
**Acordo válido:** `ID_REC_STATUS IN (1, 3, 12)` e `PARCELA = 0`  
**Batimento:** evento `QTD_NV_CLI > 10.000` em `CARGA_LOTE` onde `ID_USUARIO = 1`

### Faixas de batimento

| Faixa | Critério | Dias no dataset |
|---|---|---|
| `pos_batimento` | 0–5 dias desde último batimento | 22 dias |
| `absorcao` | 6–15 dias | 28 dias |
| `basal` | > 15 dias | 80 dias |

---

## 2. Dataset (`d.csv`)

### Schema

| Coluna | Tipo | Notas |
|---|---|---|
| `banco_origem` | string | `AUTOS` ou `CONSUMER` |
| `dia` | date | Dias úteis somente |
| `hora` | int | 8–19 |
| `dia_semana` | int | 2=seg … 6=sex |
| `total_dia` | int | Total de acordos no dia (conhecido no fim do dia) |
| `acumulado_ate_hora` | int | Acordos acumulados até essa hora |
| `proporcao_ate_hora` | float | `acumulado / total_dia` |
| `data_ultimo_batimento` | date | NULL para 45 dias basais antigos |
| `dias_desde_ultimo_batimento` | int | NULL para os mesmos 45 dias → imputado |
| `faixa_batimento` | string | `pos_batimento` / `absorcao` / `basal` |

### Estatísticas validadas

- **Linhas:** 3.144 · **Dias distintos:** 131 · **Período:** 2025-11-07 → 2026-05-08  
- Dois bancos incluídos (CONSUMER + AUTOS)
- NULLs em `data_ultimo_batimento` / `dias_desde_ultimo_batimento`: 446 linhas (basais antigos sem registro de batimento)

### Pré-processamento aplicado

| Etapa | Detalhe |
|---|---|
| `dia` | `pd.to_datetime` |
| `acumulado_ate_hora` | `pd.to_numeric(errors='coerce')` |
| `dias_desde_ultimo_batimento` | `pd.to_numeric` + **imputação: NULL → 30** (basal, > 15 dias, conservador) |
| `acordos_banda` | `diff()` de `acumulado_ate_hora` por `(banco_origem, dia)`, preenchido com `acumulado` na primeira hora; negativos clipados a 0 |
| Separador CSV | `;` · Decimal `,` |

---

## 3. Modelo Phase 1 — Mediana por Banda (MVP)

### Descrição

Tabela de lookup: `(hora, faixa_batimento) → mediana histórica de acordos_banda`

### Justificativa da mediana

Outliers confirmados no dataset (ex: `pos_batimento` 17h tem mean=6.3, median=5.0). Mediana é robusta a dias atípicos. Média foi descartada explicitamente.

### Split de validação

| Conjunto | Dias | Período |
|---|---|---|
| Treino | 100 | 2025-11-07 → 2026-03-26 |
| Holdout | 30 | 2026-03-27 → 2026-05-07 |

### Métricas no holdout (30 dias) — projeção de `total_dia`

> **Atenção:** estas métricas medem a projeção do fechamento do dia (`acumulado / proporcao_mediana`), não o `esperado` por banda.

| Métrica | Valor |
|---|---|
| MAE (acordos/dia) | **9.85** |
| MAPE | **22.6%** |
| N observações | 617 |

**MAE por faixa (projeção total_dia):**

| Faixa | MAE | MAPE | n |
|---|---|---|---|
| absorcao | 9.27 | 22.3% | 145 |
| basal | 9.44 | 23.1% | 317 |
| pos_batimento | 11.25 | 21.8% | 155 |

> MAPE acima do threshold de 20% em todas as faixas. O problema não é a granularidade das faixas — é o modelo de projeção em si. Ver seção 5 (testes de simplificação).

### Métricas no holdout (20 dias) — `acordos_banda` por hora

> Estas medem o `esperado` por banda horária (output direto do modelo).

| Métrica | Valor |
|---|---|
| MAE (acordos/banda) | **3.07** |
| N observações | 412 |

**MAE por faixa:**

| Faixa | MAE | n |
|---|---|---|
| absorcao | 3.43 | 145 |
| basal | 2.76 | 159 |
| pos_batimento | 3.06 | 108 |

---

## 4. Modelo Phase 2 — KNN (sklearn 1.8.0)

### Arquitetura

**Features (7):** `hora`, `dias_desde_ultimo_batimento`, `dia_semana_sin`, `dia_semana_cos`, `acumulado_lag`, `banco_bin`, `acumulado_primeiras_2h`  
**Target:** `acordos_banda` (incremental por hora)  
**Métrica de distância:** Euclidiana  
**Normalização:** `StandardScaler` (ajustado somente no treino, nunca no serving)  
**Fallback:** Phase 1 median lookup `(hora, faixa_batimento)` quando vizinhos insuficientes

**Notas das features novas:**
- `banco_bin` — 0=AUTOS, 1=CONSUMER (binária)
- `acumulado_primeiras_2h` — soma de `acordos_banda` das horas 8+9 do dia (por banco). Zero para `hora < 10` (não disponível na hora de prever). Sem leakage.
- `acumulado_lag` — acumulado até a hora anterior (shift(1) dentro do grupo banco+dia)
- `dia_semana_sin/cos` — codificação cíclica (2=seg…6=sex, período=5)

### Parâmetros do modelo selecionado

| Parâmetro | Valor |
|---|---|
| `k` | 20 |
| `metric` | euclidean |
| Features | 7 (ver acima) |
| sklearn | 1.8.0 |

### Seleção de k — Walk-forward CV (folds 1–3, fold 4 é holdout)

`K_OPTS = [5, 7, 10, 15, 20]`  
Folds (sobre 131 dias): treino[0:91/101/111/121] · teste[91-101/101-111/111-121/121-131]

| k | Fold1 | Fold2 | Fold3 | MAE médio |
|---|---|---|---|---|
| 5 | 0.55 | 0.93 | 0.81 | 0.76 |
| 7 | 0.55 | 0.86 | 0.75 | 0.72 |
| 10 | 0.52 | 0.82 | 0.72 | 0.69 |
| 15 | 0.52 | 0.81 | 0.71 | 0.68 |
| **20** | **0.53** | **0.78** | **0.68** | **0.66** |

### Split de validação

| Conjunto | Dias | Período |
|---|---|---|
| Treino (fold 4) | 121 | 2025-11-07 → 2026-04-24 |
| Holdout (fold 4) | 10 | 2026-04-27 → 2026-05-08 |
| Treino deploy | 131 | 2025-11-07 → 2026-05-08 (modelo final salvo) |

### Métricas no holdout (Fold 4) — Phase 1 vs KNN k=20

| Métrica | Phase 1 | KNN k=20 | Delta |
|---|---|---|---|
| **MAE (acordos/banda)** | **1.28** | **1.20** | **−0.09 (−6.8%)** |
| Acerto direcional | — | 57.3% | — |
| N obs | 240 | 240 | — |

**MAE por faixa (fold 4):**

| Faixa | Phase 1 | KNN | Delta | n |
|---|---|---|---|---|
| absorcao | 1.12 | 1.15 | +0.03 | 60 |
| basal | 1.64 | 1.45 | −0.20 | 132 |
| pos_batimento | 0.50 | 0.56 | +0.06 | 48 |

**MAE por hora (fold 4):**

| Hora | Phase 1 | KNN | Delta | n |
|---|---|---|---|---|
| 8 | 0.45 | 0.50 | +0.05 | 20 |
| 9 | 1.35 | 1.40 | +0.05 | 20 |
| 10 | 1.80 | 1.55 | −0.25 | 20 |
| 11 | 2.00 | 1.75 | −0.25 | 20 |
| 12 | 1.05 | 1.05 | 0.00 | 20 |
| 13 | 2.10 | 1.95 | −0.15 | 20 |
| 14 | 2.50 | 2.10 | −0.40 | 20 |
| 15 | 1.85 | 1.45 | −0.40 | 20 |
| 16 | 1.15 | 0.85 | −0.30 | 20 |
| 17 | 1.05 | 0.80 | −0.25 | 20 |
| 18 | 0.10 | 0.55 | **+0.45** | 20 |
| 19 | 0.00 | 0.40 | **+0.40** | 20 |

### Resumo dos 4 folds (k=20)

| Fold | Phase 1 | KNN k=20 | Delta | Status |
|---|---|---|---|---|
| 1 | 0.46 | 0.53 | +0.07 | PIOR |
| 2 | 0.69 | 0.78 | +0.08 | PIOR |
| 3 | 0.66 | 0.68 | +0.02 | PIOR |
| 4 | 1.28 | 1.20 | −0.09 | OK |
| **Média** | **0.78** | **0.79** | **+0.02** | — |

### Histórico de experimentos (sklearn 1.8)

| Config | MAE Fold 4 | Δ vs P1 | Direcional | MAE médio 4f |
|---|---|---|---|---|
| Base k=10 (5 features) | 1.27 | −1.3% | 47.3% | 0.86 |
| +banco_bin k=10 | 1.21 | −5.5% | 59.1% | 0.84 |
| +primeiras_2h k=10 | 1.19 | −7.1% | 56.4% | 0.82 |
| **+primeiras_2h k=20** | **1.20** | **−6.8%** | **57.3%** | **0.79** |

CV de folds 1-3 favoreceu k=20 (menor MAE médio). Em fold 4 isolado k=10 vence por 0.01.

### Critérios de promoção (todos exigidos)

| # | Critério | Resultado | Status |
|---|---|---|---|
| 1 | MAE < 8.5 | 1.20 | OK |
| 2 | Melhora ≥ 10% vs Phase 1 | −6.8% | FALHOU |
| 3 | Acerto direcional > 65% | 57.3% | FALHOU |
| 4 | Nenhum fold pior que Phase 1 | folds 1,2,3 piores | FALHOU |

### Decisão de promoção

**RESULTADO: MANTER Phase 1.** Três critérios de promoção falharam.

**Análise:**
- Critério 2 ficou a 3.2 pp do alvo (−6.8% vs −10% exigido). Melhor que iteração anterior (−1.3% sem novas features).
- Critério 3 (direcional) continua abaixo de 65% — modelo acerta magnitude mas não consegue prever consistentemente se a hora vai render mais ou menos que a mediana.
- Folds 1-3 operam em regime de volume baixo (MAE Phase 1 ≤ 0.69, mediana de `acordos_banda` é 0). Phase 1 vence porque prever "0" é difícil de bater. KNN só ganha em fold 4 quando há volume suficiente para diferenciar contexto.

**Status atual: Phase 1 em produção.** Artefatos `knn_phase2_*.joblib` salvos (treinados em 131 dias) ficam disponíveis para A/B futuro caso o threshold seja revisado.

---

## 4b. Model B — KNN para `projecao_fechamento` (total_dia)

### O que foi testado

Dois experimentos para verificar se KNN k=10 supera Phase 1 na projeção do total de acordos do dia.

**Experimento v1** (`knn_modelB_total_dia.py`): `hora` usada como filtro, features 2D `[dias, acum]`.

**Experimento v2** (`knn_modelB_v2.py`): `hora` incluída no vetor de features 3D `[hora_norm, dias_norm, acum_norm]`, sem filtro por hora ou faixa.

### Split

100 dias treino / 30 dias holdout (2026-03-27 → 2026-05-07) — espelho exato de `validacao_faixas.py` para comparar com baseline 9.85.

### Métricas no holdout

| Métrica | Phase 1 | KNN v1 (hora=filtro) | KNN v2 (hora=feature) |
|---|---|---|---|
| **MAE (acordos)** | **9.85** | 20.06 | 14.93 |
| N obs | 617 | 617 | 617 |

| Faixa | Phase 1 | KNN v1 | KNN v2 |
|---|---|---|---|
| absorcao | 9.27 | 28.77 | 20.20 |
| basal | 9.44 | 14.35 | 12.13 |
| pos_batimento | 11.25 | 23.60 | 15.71 |

A correção de v1 → v2 melhorou +5.13 acordos de MAE global e atenuou o padrão de piora crescente com a hora. Ainda assim, KNN perde para Phase 1 em todas as faixas e em 11 de 12 horas.

### Por que KNN não pode ganhar de Phase 1 em `total_dia`

O KNN falhou não por ser um modelo inadequado — mas porque a `proporcao_ate_hora` já encapsula exatamente a relação que ele tentaria aprender.

Por definição:

```
proporcao_ate_hora = acumulado_ate_hora / total_dia
```

Phase 1 inverte essa identidade diretamente:

```
projecao = acumulado_ate_hora / proporcao_mediana
```

O KNN tentou estimar `total_dia` por similaridade de contexto `(hora, dias_desde_batimento, acumulado)`. Mas ao fazer isso, está aproximando indiretamente uma relação que Phase 1 expressa como divisão direta da mesma grandeza. **Um modelo de similaridade não tem vantagem de informação sobre a fórmula que usa o dado diretamente.**

Para o KNN ganhar em `total_dia`, precisaria de features que Phase 1 ignora — como padrão intra-dia, sazonalidade, ou eventos externos. Com as features disponíveis, a competição é assimétrica.

### Decisão final — Model B

**Todos os critérios falharam nos dois experimentos. MANTER Phase 1 para `projecao_fechamento`.**

| Critério | v1 | v2 |
|---|---|---|
| MAE < 9.85 | FALHOU (20.06) | FALHOU (14.93) |
| Melhora ≥ 5% | FALHOU (+103.7%) | FALHOU (+51.5%) |
| Melhora em ≥ 2/3 faixas | FALHOU (0/3) | FALHOU (0/3) |

**Escopo definitivo do KNN Phase 2:**
- `bandas[].esperado` (`acordos_banda`) → **KNN k=10** ✅
- `projecao_fechamento` (`total_dia`) → **Phase 1 (fórmula de proporção)** ✅

---

## 5. Testes de Simplificação e Experimentos Adicionais

### 5.1 Simplificação de faixas: 3 → 2 (`pos_reshuffle`)

**Hipótese:** fundir `pos_batimento` + `absorcao` em `pos_reshuffle` sem degradação.

**Resultado:** 2 faixas tem MAE marginalmente melhor (9.74 vs 9.85) e MAPE menor (22.4% vs 22.6%). Diferença estatisticamente **não significativa** (p=0.177, teste t pareado).

**Conclusão:** simplificação é tecnicamente viável, mas ambos os modelos ficam fora do threshold de MAPE < 20%. O problema é o modelo de projeção, não a granularidade das faixas. **Manter 3 faixas** para preservar sinais distintos de negócio.

### 5.2 Adição de `dia_semana` ao lookup (3D)

**Hipótese:** `(hora, faixa, dia_semana)` captura padrões semanais e reduz MAE.

**Resultado:**

| Modelo | MAE | MAPE |
|---|---|---|
| 2D — (hora, faixa) | 9.85 | 22.6% |
| 3D — (hora, faixa, dia_semana) | **11.17** | **26.1%** |

O modelo 3D é **significativamente pior** (p=0.0001, +13.4% MAE). 67% das células têm menos de 10 observações (mediana = 7 obs/célula). Medianas instáveis causam overfitting.

**Conclusão:** `dia_semana` não deve ser adicionado ao lookup com o dataset atual. Revisar quando houver ≥ 3 anos de histórico por faixa (~750+ dias).

### 5.3 Janela Abril — modelo sem `faixa_batimento`

**Dataset:** `dia >= 2026-04-07` (23 dias, 480 linhas). Split: 17 treino / 6 holdout.

**Hipótese:** na janela de 23 dias com ciclo de batimento incompleto (batimento 29/04 ainda ativo), `faixa_batimento` adiciona ruído. Mediana sem faixa usa ~3× mais observações por célula → mais estável.

**Baseline (3 faixas, janela abril):** MAE 16.03, MAPE 39.5%

| Faixa (baseline) | MAE | n |
|---|---|---|
| absorcao | 27.07 | 27 |
| basal | 8.59 | 57 |
| pos_batimento | 19.74 | 34 |

**Test A — Phase 1 sem faixa** (`hora → mediana proporcao`):

| Métrica | Baseline | Test A | Delta |
|---|---|---|---|
| MAE global | 16.03 | **13.36** | **−2.67 (−16.6%)** |
| N obs | 118 | 118 | — |

Test A melhora em 9 de 12 horas. Maior ganho nas horas iniciais (8h: −16.74, 10h: −6.15). Leve piora em 16h (+0.97).

**Test B — KNN 2D `[hora_norm, acum_norm]`:** MAE 77.36 (+382%). Catastrófico — sem `dias_desde_batimento`, vizinhos por `(hora, acum)` têm `total_dia` completamente diferente. Descartado.

**Conclusão — janela abril:** `faixa_batimento` é ruído com < 30 dias e ciclo incompleto. Test A promovido para a janela abril: lookup `hora → mediana proporcao`, MAE 13.36 vs threshold 15.23.

**Regra de chaveamento:** usar lookup sem faixa quando `dias_em_janela < 30` ou quando o batimento mais recente tem < 16 dias (ciclo pos+absorcao não completo). Retornar ao modelo com 3 faixas quando a janela crescer.

---

## 6. Escolhas Técnicas

| Decisão | Escolha | Alternativa descartada | Motivo |
|---|---|---|---|
| Métrica central | MAE | MAPE | MAPE distorce com denominadores 2–3 acordos |
| Tendência central | Mediana | Média | Outliers confirmados em todas as faixas |
| Validação temporal | Walk-forward | K-fold aleatório | Série temporal — shuffle vaza futuro para treino |
| Normalização KNN | `StandardScaler` | Min-Max manual | Mais robusto a outliers; integração nativa sklearn |
| Distância KNN | Euclidiana | Coseno, Manhattan | Features contínuas em plano 2D; proximidade geométrica é o que importa |
| Impute NULL dias | 30 | Mediana, moda | Basal = > 15 dias; 30 é conservador e não invade outra faixa |
| k mínimo produção | 5 | < 5 | Previne memorização de dias individuais |

---

## 7. Qualidade dos Dados

| Verificação | Status | Detalhe |
|---|---|---|
| Negativos em `acordos_banda` | ✅ Zero | Nenhuma inconsistência no diff |
| NULLs em `dias_desde_ultimo_batimento` | ⚠️ 446 imputados | Basais antigos sem batimento registrado — não afetam faixa |
| NULLs em `data_ultimo_batimento` | ⚠️ Presentes | Apenas nos mesmos 45 dias basais — descartados do modelo |
| Cobertura de células 2D (hora × faixa) | ✅ 36/36 | Todas as células com dados no treino |
| Cobertura de células 3D (hora × faixa × dia) | ⚠️ 16/180 < 5 obs | Impede uso de `dia_semana` como dimensão |
| Consistência temporal | ✅ | Nenhum fim de semana no dataset; horas 8–19 apenas |

---

## 8. Integrações e Serving

### Padrão de cache (FastAPI)

```python
_LOOKUP_CACHE: dict | None = None
_LOOKUP_UPDATED: datetime | None = None
_LOOKUP_TTL: int = 86400  # recalcula 1x/dia
```

### Faixa em tempo real

```python
ultimo_batimento = query_ultimo_batimento(db)
dias_desde = (date.today() - ultimo_batimento).days
faixa = 'pos_batimento' if dias_desde <= 5 else 'absorcao' if dias_desde <= 15 else 'basal'
```

### Projeção de fechamento (Phase 1)

```python
proporcao = lookup_proporcao[(hora_atual, faixa)]
projecao  = round(acumulado_hoje / proporcao) if proporcao > 0 else None
```

### Endpoint

```
GET /dashboard/ritmo-dia/{db}
db: COBwebRCBCONSUMER | COBwebRCBAUTOS | todos
```

---

## 9. Roadmap

| Output da API | Modelo em produção | Modelo Phase 2 | Status |
|---|---|---|---|
| `bandas[].esperado` | Phase 1 — mediana (hora × faixa) | KNN k=20 (7 features) | ❌ MANTER P1 (−6.8% MAE, 3/4 critérios falharam) |
| `projecao_fechamento` | Phase 1 — fórmula de proporção | KNN testado e descartado | ✅ Phase 1 definitivo |

### Pendências antes de promover KNN para `bandas[].esperado`

- [ ] Reavaliar threshold de promoção — atual exige -10% MAE + 65% direcional; nenhum modelo testado bateu ambos
- [ ] Investigar piora em folds 1-3 (regime de baixo volume) — talvez chavear: Phase 1 quando volume<X, KNN caso contrário
- [ ] Aumentar janela de treino além de 131 dias para reduzir ruído em direcional
- [ ] Implementar OOD warning em produção quando features fora do range de treino

### Fase 3 — Regressão Ridge + MLflow

Entra apenas se KNN for promovido e seu MAE ainda não satisfizer. Walk-forward obrigatório. Comparar contra KNN antes de promover.

---

## 10. Scripts de Validação

| Arquivo | Target | O que faz |
|---|---|---|
| `validacao_faixas.py` | `total_dia` | Compara 3 faixas vs 2 faixas (MAE, MAPE, teste t) |
| `validacao_dia_semana.py` | `total_dia` | Testa `dia_semana` como 3ª dimensão no lookup |
| `knn_phase2.py` | `acordos_banda` | Seleção de k (walk-forward CV) + avaliação KNN Model A |
| `knn_modelB_total_dia.py` | `total_dia` | KNN v1 com hora como filtro — descartado (MAE 20.06) |
| `knn_modelB_v2.py` | `total_dia` | KNN v2 com hora como feature 3D — descartado (MAE 14.93) |
| `teste_sem_faixa_abril.py` | `total_dia` | Janela abril: Test A (sem faixa, MAE 13.36 ✅) · Test B (KNN 2D, descartado) |
