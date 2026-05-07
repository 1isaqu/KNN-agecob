"""
KNN Phase 2 -- Model B: projecao_fechamento (total_dia)
Compara KNN k=10 vs Phase 1 no mesmo holdout de 30 dias de validacao_faixas.py.
k fixo em 10 -- sem re-selecao.
"""
import pandas as pd
import numpy as np

SEP = '-' * 54

# ---- STEP 0: INSPECAO DO DATASET ----------------------------------------
print('=' * 62)
print('STEP 0 -- INSPECAO DO DATASET')
print('=' * 62)

# Adaptacao: arquivo tem banco_origem como 1a coluna (adicionado pos pipeline v3).
# Prompt original lista 9 colunas; arquivo real tem 10.
COLS = ['banco_origem','dia','hora','dia_semana','total_dia',
        'acumulado_ate_hora','proporcao_ate_hora',
        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento']

df = pd.read_csv('d.csv', sep=';', decimal=',', names=COLS)

print('Colunas e dtypes:')
for col in df.columns:
    print('  {:<35} {}'.format(col, df[col].dtype))
print()
print('3 amostras:')
print(df[['banco_origem','dia','hora','total_dia','acumulado_ate_hora',
          'proporcao_ate_hora','dias_desde_ultimo_batimento',
          'faixa_batimento']].head(3).to_string())
print()

df['dia'] = pd.to_datetime(df['dia'])
df['total_dia'] = pd.to_numeric(df['total_dia'], errors='coerce')
df['acumulado_ate_hora'] = pd.to_numeric(df['acumulado_ate_hora'], errors='coerce')
df['dias_desde_ultimo_batimento'] = pd.to_numeric(
    df['dias_desde_ultimo_batimento'], errors='coerce')

print('total_dia       numerico: {}  nulls: {}'.format(
    pd.api.types.is_numeric_dtype(df['total_dia']),
    df['total_dia'].isna().sum()))
print('acumulado       numerico: {}  nulls: {}'.format(
    pd.api.types.is_numeric_dtype(df['acumulado_ate_hora']),
    df['acumulado_ate_hora'].isna().sum()))

n_null = df['dias_desde_ultimo_batimento'].isna().sum()
df['dias_desde_ultimo_batimento'] = df['dias_desde_ultimo_batimento'].fillna(30)
print('NULLs dias_desde_ultimo_batimento imputados com 30: {}'.format(n_null))

# ---- SPLIT TEMPORAL ------------------------------------------------------
print()
print('=' * 62)
print('SPLIT -- 100 treino / 30 holdout (espelho de validacao_faixas.py)')
print('=' * 62)

dias_ord     = sorted(df['dia'].unique())
TRAIN_DAYS   = 100   # 130 dias totais: 100 + 30 = 130, sem sobreposicao
HOLDOUT_DAYS = 30

train_dias   = set(dias_ord[:TRAIN_DAYS])
holdout_dias = set(dias_ord[-HOLDOUT_DAYS:])
assert not (train_dias & holdout_dias), 'Vazamento treino/holdout!'

train   = df[df['dia'].isin(train_dias)].copy().reset_index(drop=True)
holdout = df[df['dia'].isin(holdout_dias)].copy().reset_index(drop=True)

print('Treino:  {} dias ({} linhas)  {} -> {}'.format(
    len(train_dias), len(train),
    pd.Timestamp(min(train_dias)).date(),
    pd.Timestamp(max(train_dias)).date()))
print('Holdout: {} dias ({} linhas)  {} -> {}'.format(
    len(holdout_dias), len(holdout),
    pd.Timestamp(min(holdout_dias)).date(),
    pd.Timestamp(max(holdout_dias)).date()))

# ---- NORMALIZACAO min-max (ajustada somente no treino) -------------------
MIN_DIAS = train['dias_desde_ultimo_batimento'].min()
MAX_DIAS = train['dias_desde_ultimo_batimento'].max()
MIN_ACUM = train['acumulado_ate_hora'].min()
MAX_ACUM = train['acumulado_ate_hora'].max()

print()
print('Scaler (treino):')
print('  dias_desde_batimento: [{}, {}]'.format(int(MIN_DIAS), int(MAX_DIAS)))
print('  acumulado_ate_hora:   [{}, {}]'.format(int(MIN_ACUM), int(MAX_ACUM)))

ood_count = [0]

def normalize(dias, acum):
    d = (dias - MIN_DIAS) / (MAX_DIAS - MIN_DIAS)
    a = (acum - MIN_ACUM) / (MAX_ACUM - MIN_ACUM)
    if d > 1.0 or d < 0.0 or a > 1.0 or a < 0.0:
        ood_count[0] += 1
    return max(0.0, min(1.0, d)), max(0.0, min(1.0, a))

train['dias_norm'] = train['dias_desde_ultimo_batimento'].apply(
    lambda x: max(0.0, min(1.0, (x - MIN_DIAS) / (MAX_DIAS - MIN_DIAS))))
train['acum_norm'] = train['acumulado_ate_hora'].apply(
    lambda x: max(0.0, min(1.0, (x - MIN_ACUM) / (MAX_ACUM - MIN_ACUM))))

# ---- PHASE 1 BASELINE (proporcao lookup) ---------------------------------
p1_lookup = (train
    .groupby(['hora','faixa_batimento'])['proporcao_ate_hora']
    .median()
    .to_dict())

def phase1_predict(hora, faixa, acumulado):
    prop = p1_lookup.get((hora, faixa))
    if prop and prop > 0:
        return acumulado / prop
    return np.nan

# ---- KNN MODEL B (vizinhos -> mediana de total_dia) ---------------------
K = 10
fallback_count = [0]

train_idx = {}
for (h, f), grp in train.groupby(['hora','faixa_batimento']):
    train_idx[(h, f)] = grp

def knn_predict(hora, faixa, dias_q, acum_q):
    cands = train_idx.get((hora, faixa))
    if cands is None or len(cands) < K:
        fallback_count[0] += 1
        return phase1_predict(hora, faixa, acum_q)
    dq, aq = normalize(dias_q, acum_q)
    d_arr = cands['dias_norm'].values
    a_arr = cands['acum_norm'].values
    dists = np.sqrt((d_arr - dq)**2 + (a_arr - aq)**2)
    k_idx = np.argpartition(dists, K)[:K]
    return float(np.median(cands['total_dia'].values[k_idx]))

# ---- AVALIACAO NO HOLDOUT ------------------------------------------------
print()
print('=' * 62)
print('AVALIACAO NO HOLDOUT (k=10, target=total_dia)')
print('=' * 62)

rows_h = list(holdout.iterrows())

p1_preds  = np.array([phase1_predict(
    r['hora'], r['faixa_batimento'], r['acumulado_ate_hora'])
    for _, r in rows_h], dtype=float)

knn_preds = np.array([knn_predict(
    r['hora'], r['faixa_batimento'],
    r['dias_desde_ultimo_batimento'], r['acumulado_ate_hora'])
    for _, r in rows_h], dtype=float)

real = holdout['total_dia'].values.astype(float)

print('OOD clipping ativado:  {}'.format(ood_count[0]))
print('Fallbacks KNN -> P1:   {}'.format(fallback_count[0]))

mask  = ~(np.isnan(p1_preds) | np.isnan(knn_preds) | np.isnan(real))
p1_v  = p1_preds[mask]
knn_v = knn_preds[mask]
real_v = real[mask]

mae_p1  = float(np.abs(p1_v  - real_v).mean())
mae_knn = float(np.abs(knn_v - real_v).mean())
delta   = mae_knn - mae_p1

print()
print('{:<16} {:>9} {:>10} {:>8}'.format('Metrica', 'Phase 1', 'KNN k=10', 'Delta'))
print(SEP)
print('{:<16} {:>9.2f} {:>10.2f} {:>+8.2f}'.format('MAE (acordos)', mae_p1, mae_knn, delta))
print('{:<16} {:>9} {:>10} {:>8}'.format('N obs', int(mask.sum()), int(mask.sum()), '--'))

# Estratificacao por faixa
holdout_v = holdout[mask].copy().reset_index(drop=True)
holdout_v['p1_pred']  = p1_v
holdout_v['knn_pred'] = knn_v
holdout_v['err_p1']   = np.abs(p1_v  - real_v)
holdout_v['err_knn']  = np.abs(knn_v - real_v)

print()
print('MAE por faixa:')
print('{:<16} {:>9} {:>10} {:>8}'.format('Faixa', 'Phase 1', 'KNN k=10', 'Delta'))
print(SEP)
faixa_results = {}
for f in ['absorcao','basal','pos_batimento']:
    sub = holdout_v[holdout_v['faixa_batimento'] == f]
    if len(sub) == 0:
        continue
    m1 = float(sub['err_p1'].mean())
    mk = float(sub['err_knn'].mean())
    d  = mk - m1
    faixa_results[f] = d
    print('{:<16} {:>9.2f} {:>10.2f} {:>+8.2f}  n={}'.format(f, m1, mk, d, len(sub)))

faixas_melhores = sum(1 for d in faixa_results.values() if d < 0)

# MAE por hora
print()
print('MAE por hora (8h-19h):')
print('{:<6} {:>9} {:>10} {:>8}'.format('Hora', 'Phase 1', 'KNN k=10', 'Delta'))
print(SEP)
horas_piores = []
for h in sorted(holdout_v['hora'].unique()):
    sub = holdout_v[holdout_v['hora'] == h]
    m1 = float(sub['err_p1'].mean())
    mk = float(sub['err_knn'].mean())
    d  = mk - m1
    flag = ' <- pior' if d > 0.5 else ''
    if d > 0:
        horas_piores.append(h)
    print('{:<6} {:>9.2f} {:>10.2f} {:>+8.2f}  n={}{}'.format(
        h, m1, mk, d, len(sub), flag))

# ---- DECISAO ------------------------------------------------------------
print()
print('=' * 62)
print('DECISAO DE PROMOCAO -- Model B (projecao_fechamento)')
print('=' * 62)

delta_pct    = delta / mae_p1 * 100
crit_mae     = mae_knn < 9.85
crit_melhora = delta_pct <= -5.0
crit_faixas  = faixas_melhores >= 2

print('Criterio 1 -- MAE < 9.85:                {} (MAE={:.2f})'.format(
    'OK' if crit_mae else 'FALHOU', mae_knn))
print('Criterio 2 -- Melhora >= 5% ({:.1f}%):   {}'.format(
    abs(delta_pct), 'OK' if crit_melhora else 'FALHOU'))
print('Criterio 3 -- Melhora em >= 2/3 faixas:  {} ({}/3)'.format(
    'OK' if crit_faixas else 'FALHOU', faixas_melhores))
print()

if crit_mae and crit_melhora and crit_faixas:
    print('RESULTADO: PROMOVER KNN para projecao_fechamento.')
    print('  Delta: {:+.2f} acordos ({:+.1f}%)'.format(delta, delta_pct))
    if horas_piores:
        print('  Horas com degradacao (supressao frontend recomendada):',
              ', '.join('{}h'.format(h) for h in sorted(horas_piores)))
else:
    print('RESULTADO: MANTER Phase 1 para projecao_fechamento.')
    if not crit_mae:
        print('  FALHOU: MAE {:.2f} nao supera baseline 9.85'.format(mae_knn))
    if not crit_melhora:
        print('  FALHOU: melhora {:.1f}% < threshold 5%'.format(abs(delta_pct)))
    if not crit_faixas:
        print('  FALHOU: melhora em apenas {}/3 faixas'.format(faixas_melhores))
    print()
    print('  Evidencia: KNN adiciona valor para band esperado (Model A)')
    print('  mas nao para projecao de fechamento de dia (Model B).')
