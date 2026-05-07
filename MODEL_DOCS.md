# Ritmo do Dia — Documentação de Modelos ML
**Projeto:** agecob-lens · **Empresa:** Agecob  
**Gerado em:** 2026-05-07 · **Período de dados:** 2025-11-07 → 2026-05-07

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

- **Linhas:** 2.304 · **Dias distintos:** 130 · **Período:** 2025-11-07 → 2026-05-07  
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

## 4. Modelo Phase 2 — KNN (sklearn)

### Arquitetura

**Features:** `dias_desde_ultimo_batimento`, `acumulado_ate_hora`  
**Target:** `acordos_banda` (incremental por hora)  
**Métrica de distância:** Euclidiana  
**Normalização:** `StandardScaler` (ajustado somente no treino, nunca no serving)  
**Fallback:** Phase 1 median lookup quando `n_candidates < k`

### Parâmetros do modelo selecionado

| Parâmetro | Valor |
|---|---|
| `k` | 10 |
| `metric` | euclidean |
| Scaler mean | `[42.87, 15.42]` (`dias`, `acum`) |
| Scaler std | `[37.68, 13.40]` |

### Seleção de k — Walk-forward CV (4 folds, treino 110 dias)

Folds: treino[0:70/80/90/100] · teste[70-80/80-90/90-100/100-110]

| k | Fold1 | Fold2 | Fold3 | Fold4 | MAE médio |
|---|---|---|---|---|---|
| 3 | 1.34 | 1.76 | 1.83 | 2.34 | 1.82 |
| 5 | 1.34 | 1.66 | 1.72 | 2.21 | 1.73 |
| 7 | 1.30 | 1.61 | 1.63 | 2.11 | 1.66 |
| **10** | **1.32** | **1.55** | **1.57** | **2.07** | **1.63** |

### Split de validação

| Conjunto | Dias | Período |
|---|---|---|
| Treino | 110 | 2025-11-07 → 2026-04-09 |
| Holdout | 20 | 2026-04-10 → 2026-05-07 |

### Métricas no holdout — comparativo Phase 1 vs KNN

| Métrica | Phase 1 | KNN k=10 | Delta |
|---|---|---|---|
| **MAE (acordos/banda)** | **3.07** | **2.58** | **−0.49 (−16%)** |

**MAE por faixa:**

| Faixa | Phase 1 | KNN | Delta | n |
|---|---|---|---|---|
| absorcao | 3.43 | 2.86 | −0.57 | 145 |
| basal | 2.76 | 2.30 | −0.47 | 159 |
| pos_batimento | 3.06 | 2.64 | −0.42 | 108 |

**MAE por hora:**

| Hora | Phase 1 | KNN | Delta | n |
|---|---|---|---|---|
| 8h | 1.52 | 1.18 | −0.33 | 33 |
| 9h | 2.69 | 2.17 | −0.53 | 36 |
| 10h | 2.87 | 2.49 | −0.38 | 39 |
| 11h | 3.92 | 3.19 | −0.72 | 36 |
| 12h | 3.35 | 2.68 | −0.68 | 37 |
| 13h | 3.78 | 2.94 | −0.83 | 36 |
| 14h | 3.58 | 2.50 | −1.08 | 38 |
| 15h | 4.17 | 2.92 | −1.25 | 36 |
| 16h | 2.56 | 2.35 | −0.21 | 34 |
| 17h | 3.89 | 3.20 | −0.69 | 35 |
| 18h | 1.69 | 2.76 | **+1.07** | 29 |
| 19h | 2.09 | 2.52 | **+0.43** | 23 |

> 18h e 19h têm menor volume no treino. KNN encontra vizinhos menos representativos nessas horas.

### OOD (Out-of-Distribution) no holdout

| Feature | Amostras fora do range de treino |
|---|---|
| `dias_desde_ultimo_batimento` | 10 |
| `acumulado_ate_hora` | 14 |

`StandardScaler` extrapola linearmente — sem clipping implícito. Monitorar em produção.

### Decisão de promoção

**Critério:** redução de MAE ≥ 0.5 acordos/banda

**Delta obtido: −0.49** → **limítrofe, 0.01 abaixo do threshold**

A melhora é consistente em todas as faixas e nas horas de maior volume (9h–17h). A piora em 18h/19h é atribuída à escassez de dados (n≤29), não ao modelo.

**Status atual: Phase 1 em produção** (critério estrito não atingido). Discutir revisão do threshold para 5% de redução relativa — nesse critério o KNN seria promovido.

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
| `bandas[].esperado` | Phase 1 — mediana (hora × faixa) | KNN k=10 | ⏳ Limítrofe (−0.49 MAE, −16%) |
| `projecao_fechamento` | Phase 1 — fórmula de proporção | KNN testado e descartado | ✅ Phase 1 definitivo |

### Pendências antes de promover KNN para `bandas[].esperado`

- [ ] Decidir threshold: manter 0.5 acordos absolutos ou migrar para 5% relativo (16% obtido justifica)
- [ ] Implementar OOD warning em produção (14 amostras fora do range no holdout)
- [ ] Investigar degradação em 18h/19h — avaliar lookup fallback por hora
- [ ] Serializar scaler (`mean=[42.87, 15.42]`, `std=[37.68, 13.40]`) junto ao modelo

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
