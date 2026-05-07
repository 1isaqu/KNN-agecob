"""
KNN Phase 2 -- Model B v2: projecao_fechamento com hora como feature de distancia.
Correcao: hora sai do filtro e entra no vetor de features 3D.
Sem filtro por hora ou faixa -- distancia separa naturalmente.
"""
import pandas as pd
import numpy as np

SEP = '-' * 60

# ---- STEP 0: INSPECAO ---------------------------------------------------
print('=' * 62)
print('STEP 0 -- INSPECAO DO DATASET')
print('=' * 62)

COLS = ['banco_origem','dia','hora','dia_semana','total_dia',
        'acumulado_ate_hora','proporcao_ate_hora',
        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento']

df = pd.read_csv('d.csv', sep=';', decimal=',', names=COLS)
df['dia'] = pd.to_datetime(df['dia'])
df['total_dia'] = pd.to_numeric(df['total_dia'], errors='coerce')
df['acumulado_ate_hora'] = pd.to_numeric(df['acumulado_ate_hora'], errors='coerce')
df['dias_desde_ultimo_batimento'] = pd.to_numeric(
    df['dias_desde_ultimo_batimento'], errors='coerce')

print('banco_origem presente: {}'.format('banco_origem' in df.columns))
print('total_dia       numerico: {}  nulls: {}'.format(
    pd.api.types.is_numeric_dtype(df['total_dia']), df['total_dia'].isna().sum()))
print('acumulado       numerico: {}  nulls: {}'.format(
    pd.api.types.is_numeric_dtype(df['acumulado_ate_hora']),
    df['acumulado_ate_hora'].isna().sum()))

n_null = df['dias_desde_ultimo_batimento'].isna().sum()
df['dias_desde_ultimo_batimento'] = df['dias_desde_ultimo_batimento'].fillna(30)
print('NULLs dias_desde_batimento imputados com 30: {}'.format(n_null))
print()
print('3 amostras:')
print(df[['banco_origem','dia','hora','total_dia',
          'acumulado_ate_hora','dias_desde_ultimo_batimento',
          'faixa_batimento']].head(3).to_string())

# ---- SPLIT --------------------------------------------------------------
print()
print('=' * 62)
print('SPLIT -- 100 treino / 30 holdout')
print('(130 dias totais: 110+30=140 excede total; ajustado para 100+30=130)')
print('=' * 62)

dias_ord     = sorted(df['dia'].unique())
TRAIN_DAYS   = 100
HOLDOUT_DAYS = 30
assert TRAIN_DAYS + HOLDOUT_DAYS == len(dias_ord), \
    'Split nao cobre exatamente todos os dias: {} + {} != {}'.format(
        TRAIN_DAYS, HOLDOUT_DAYS, len(dias_ord))

train_dias   = set(dias_ord[:TRAIN_DAYS])
holdout_dias = set(dias_ord[-HOLDOUT_DAYS:])
assert not (train_dias & holdout_dias)

train   = df[df['dia'].isin(train_dias)].copy().reset_index(drop=True)
holdout = df[df['dia'].isin(holdout_dias)].copy().reset_index(drop=True)

print('Treino:  {} dias ({} linhas)  {} -> {}'.format(
    len(train_dias), len(train),
    pd.Timestamp(min(train_dias)).date(), pd.Timestamp(max(train_dias)).date()))
print('Holdout: {} dias ({} linhas)  {} -> {}'.format(
    len(holdout_dias), len(holdout),
    pd.Timestamp(min(holdout_dias)).date(), pd.Timestamp(max(holdout_dias)).date()))

# ---- NORMALIZACAO -------------------------------------------------------
# hora: range fixo [8,19] -- sem OOD risk, sem fit nos dados
HORA_MIN, HORA_MAX = 8, 19

# dias e acum: min-max ajustado somente no treino
MIN_DIAS = float(train['dias_desde_ultimo_batimento'].min())
MAX_DIAS = float(train['dias_desde_ultimo_batimento'].max())
MIN_ACUM = float(train['acumulado_ate_hora'].min())
MAX_ACUM = float(train['acumulado_ate_hora'].max())

print()
print('Scaler (treino):')
print('  hora:                 [{}, {}]  (fixo, sem fit)'.format(HORA_MIN, HORA_MAX))
print('  dias_desde_batimento: [{:.0f}, {:.0f}]'.format(MIN_DIAS, MAX_DIAS))
print('  acumulado_ate_hora:   [{:.0f}, {:.0f}]'.format(MIN_ACUM, MAX_ACUM))

ood_dias = [0]
ood_acum = [0]

def norm_hora(h):
    return (h - HORA_MIN) / (HORA_MAX - HORA_MIN)

def norm_dias(d):
    v = (d - MIN_DIAS) / (MAX_DIAS - MIN_DIAS)
    if v < 0.0 or v > 1.0:
        ood_dias[0] += 1
    return max(0.0, min(1.0, v))

def norm_acum(a):
    v = (a - MIN_ACUM) / (MAX_ACUM - MIN_ACUM)
    if v < 0.0 or v > 1.0:
        ood_acum[0] += 1
    return max(0.0, min(1.0, v))

# Pre-normalizar treino inteiro (vetorizado)
train['hora_n'] = train['hora'].apply(norm_hora)
train['dias_n'] = train['dias_desde_ultimo_batimento'].apply(norm_dias)
train['acum_n'] = train['acumulado_ate_hora'].apply(norm_acum)

# Matriz numpy do treino para distancias vetorizadas -- muito mais rapido
X_train = train[['hora_n','dias_n','acum_n']].values   # shape (N_train, 3)
y_train = train['total_dia'].values.astype(float)       # target: total_dia

# ---- PHASE 1 BASELINE ---------------------------------------------------
p1_lookup = (train
    .groupby(['hora','faixa_batimento'])['proporcao_ate_hora']
    .median().to_dict())

def phase1_predict(hora, faixa, acumulado):
    prop = p1_lookup.get((hora, faixa))
    if prop and prop > 0:
        return acumulado / prop
    return np.nan

# ---- KNN MODEL B v2 (3D: hora+dias+acum, sem filtro) --------------------
K = 10
fallback_count = [0]

def knn_predict_v2(hora_q, faixa_q, dias_q, acum_q):
    q = np.array([norm_hora(hora_q), norm_dias(dias_q), norm_acum(acum_q)])
    # Distancias vetorizadas a todo o treino
    diffs = X_train - q                         # (N_train, 3)
    dists = np.sqrt((diffs**2).sum(axis=1))     # (N_train,)
    if len(dists) < K:
        fallback_count[0] += 1
        return phase1_predict(hora_q, faixa_q, acum_q)
    k_idx = np.argpartition(dists, K)[:K]
    return float(np.median(y_train[k_idx]))

# ---- AVALIACAO NO HOLDOUT -----------------------------------------------
print()
print('=' * 62)
print('AVALIACAO NO HOLDOUT (k=10, 3D feature: hora+dias+acum)')
print('=' * 62)

rows_h = list(holdout.iterrows())

p1_preds  = np.array([
    phase1_predict(r['hora'], r['faixa_batimento'], r['acumulado_ate_hora'])
    for _, r in rows_h], dtype=float)

knn_v2_preds = np.array([
    knn_predict_v2(r['hora'], r['faixa_batimento'],
                   r['dias_desde_ultimo_batimento'], r['acumulado_ate_hora'])
    for _, r in rows_h], dtype=float)

real = holdout['total_dia'].values.astype(float)

print('OOD dias clipping: {}  |  OOD acum clipping: {}'.format(
    ood_dias[0], ood_acum[0]))
print('Fallbacks KNN -> P1: {}'.format(fallback_count[0]))

mask   = ~(np.isnan(p1_preds) | np.isnan(knn_v2_preds) | np.isnan(real))
p1_v   = p1_preds[mask]
knn_v  = knn_v2_preds[mask]
real_v = real[mask]

mae_p1      = float(np.abs(p1_v  - real_v).mean())
mae_knn_v2  = float(np.abs(knn_v - real_v).mean())
mae_knn_v1  = 20.06  # resultado anterior para referencia
delta        = mae_knn_v2 - mae_p1

print()
print('{:<16} {:>9} {:>13} {:>14} {:>8}'.format(
    'Metrica', 'Phase 1', 'KNN v1 (bug)', 'KNN v2 (fix)', 'Delta v2'))
print(SEP)
print('{:<16} {:>9.2f} {:>13.2f} {:>14.2f} {:>+8.2f}'.format(
    'MAE (acordos)', mae_p1, mae_knn_v1, mae_knn_v2, delta))
print('{:<16} {:>9} {:>13} {:>14} {:>8}'.format(
    'N obs', int(mask.sum()), 617, int(mask.sum()), '--'))

# Estratificacao por faixa
holdout_v = holdout[mask].copy().reset_index(drop=True)
holdout_v['p1_pred']  = p1_v
holdout_v['knn_pred'] = knn_v
holdout_v['err_p1']   = np.abs(p1_v  - real_v)
holdout_v['err_knn']  = np.abs(knn_v - real_v)

# MAE v1 por faixa para comparacao
mae_v1_faixa = {'absorcao': 28.77, 'basal': 14.35, 'pos_batimento': 23.60}
mae_p1_faixa = {'absorcao': 9.27,  'basal': 9.44,  'pos_batimento': 11.25}

print()
print('MAE por faixa:')
print('{:<16} {:>9} {:>13} {:>14} {:>8}'.format(
    'Faixa', 'Phase 1', 'KNN v1 (bug)', 'KNN v2 (fix)', 'Delta v2'))
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
    print('{:<16} {:>9.2f} {:>13.2f} {:>14.2f} {:>+8.2f}  n={}'.format(
        f, m1, mae_v1_faixa.get(f, 0), mk, d, len(sub)))

faixas_melhores = sum(1 for d in faixa_results.values() if d < 0)

# MAE por hora
print()
print('MAE por hora (8h-19h):')
print('{:<6} {:>9} {:>13} {:>14} {:>8}'.format(
    'Hora', 'Phase 1', 'KNN v1 (bug)', 'KNN v2 (fix)', 'Delta v2'))
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
    print('{:<6} {:>9.2f} {:>13} {:>14.2f} {:>+8.2f}  n={}{}'.format(
        h, m1, '--', mk, d, len(sub), flag))

# ---- DECISAO ------------------------------------------------------------
print()
print('=' * 62)
print('DECISAO DE PROMOCAO -- Model B v2')
print('=' * 62)

delta_pct    = delta / mae_p1 * 100
crit_mae     = mae_knn_v2 < 9.85
crit_melhora = delta_pct <= -5.0
crit_faixas  = faixas_melhores >= 2

print('Criterio 1 -- MAE < 9.85:                {} (MAE={:.2f})'.format(
    'OK' if crit_mae else 'FALHOU', mae_knn_v2))
print('Criterio 2 -- Melhora >= 5% ({:.1f}%):   {}'.format(
    abs(delta_pct), 'OK' if crit_melhora else 'FALHOU'))
print('Criterio 3 -- Melhora em >= 2/3 faixas:  {} ({}/3)'.format(
    'OK' if crit_faixas else 'FALHOU', faixas_melhores))
print()

if crit_mae and crit_melhora and crit_faixas:
    print('RESULTADO: PROMOVER KNN v2 para projecao_fechamento.')
    print('  Delta: {:+.2f} acordos ({:+.1f}%)'.format(delta, delta_pct))
    if horas_piores:
        print('  Horas com degradacao (supressao frontend):',
              ', '.join('{}h'.format(h) for h in sorted(horas_piores)))
else:
    print('RESULTADO: MANTER Phase 1 para projecao_fechamento.')
    if not crit_mae:
        print('  FALHOU: MAE={:.2f} nao supera baseline 9.85'.format(mae_knn_v2))
    if not crit_melhora:
        print('  FALHOU: melhora {:.1f}% < threshold 5%'.format(abs(delta_pct)))
    if not crit_faixas:
        print('  FALHOU: melhora em {}/3 faixas'.format(faixas_melhores))
    print()
    print('  Conclusao final: formula de proporcao (Phase 1) e o mecanismo')
    print('  correto para projecao de fechamento de dia. KNN fica restrito')
    print('  a band esperado (Model A).')
