"""
Hipotese: na janela abril (23 dias), faixa_batimento adiciona ruido.
Test A: Phase 1 sem faixa (lookup por hora apenas).
Test B: KNN 2D [hora_norm, acum_norm] sem faixa e sem dias_desde_batimento.
Baseline: MAE 16.03 (Phase 1 com 3 faixas, holdout 6 dias).
"""
import pandas as pd
import numpy as np

SEP = '-' * 62

COLS = ['banco_origem','dia','hora','dia_semana','total_dia',
        'acumulado_ate_hora','proporcao_ate_hora',
        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento']

df = pd.read_csv('d.csv', sep=';', decimal=',', names=COLS)
df['dia'] = pd.to_datetime(df['dia'])
df['acumulado_ate_hora'] = pd.to_numeric(df['acumulado_ate_hora'], errors='coerce')
df['dias_desde_ultimo_batimento'] = pd.to_numeric(
    df['dias_desde_ultimo_batimento'], errors='coerce').fillna(30)
df = df.sort_values(['banco_origem','dia','hora']).reset_index(drop=True)
df['acordos_banda'] = (df.groupby(['banco_origem','dia'])['acumulado_ate_hora']
                         .diff().fillna(df['acumulado_ate_hora']).astype(int).clip(lower=0))

# Filtro janela abril
df = df[df['dia'] >= '2026-04-07'].copy().reset_index(drop=True)

dias_ord     = sorted(df['dia'].unique())
train_dias   = set(dias_ord[:17])
holdout_dias = set(dias_ord[17:])
train   = df[df['dia'].isin(train_dias)].copy().reset_index(drop=True)
holdout = df[df['dia'].isin(holdout_dias)].copy().reset_index(drop=True)

print('Janela: {} -> {}  |  {} dias  |  {} linhas'.format(
    df['dia'].min().date(), df['dia'].max().date(),
    df['dia'].nunique(), len(df)))
print('Treino: {} dias ({} linhas)  |  Holdout: {} dias ({} linhas)'.format(
    len(train_dias), len(train), len(holdout_dias), len(holdout)))

# ---- BASELINE: Phase 1 com faixa ----------------------------------------
p1_prop_lookup = (train.groupby(['hora','faixa_batimento'])['proporcao_ate_hora']
                       .median().to_dict())

def baseline_pred(hora, faixa, acum):
    p = p1_prop_lookup.get((hora, faixa))
    return acum / p if p and p > 0 else np.nan

base_preds = np.array([baseline_pred(r['hora'], r['faixa_batimento'],
                        r['acumulado_ate_hora'])
                       for _, r in holdout.iterrows()], dtype=float)
real = holdout['total_dia'].values.astype(float)
mask_base = ~(np.isnan(base_preds) | np.isnan(real))
mae_base  = np.abs(base_preds[mask_base] - real[mask_base]).mean()

# ---- TEST A: Phase 1 sem faixa (lookup por hora) -------------------------
prop_hora_lookup = (train.groupby('hora')['proporcao_ate_hora']
                         .median().to_dict())

def testA_pred(hora, acum):
    p = prop_hora_lookup.get(hora)
    return acum / p if p and p > 0 else np.nan

a_preds = np.array([testA_pred(r['hora'], r['acumulado_ate_hora'])
                    for _, r in holdout.iterrows()], dtype=float)
mask_a  = ~(np.isnan(a_preds) | np.isnan(real))
mae_a   = np.abs(a_preds[mask_a] - real[mask_a]).mean()

# ---- TEST B: KNN 2D [hora_norm, acum_norm] sem faixa --------------------
HORA_MIN, HORA_MAX = 8, 19
MIN_ACUM = float(train['acumulado_ate_hora'].min())
MAX_ACUM = float(train['acumulado_ate_hora'].max())
K = 10
ood_b     = [0]
fallback_b = [0]

def norm_hora(h):
    return (h - HORA_MIN) / (HORA_MAX - HORA_MIN)

def norm_acum(a):
    v = (a - MIN_ACUM) / (MAX_ACUM - MIN_ACUM)
    if v < 0.0 or v > 1.0:
        ood_b[0] += 1
    return max(0.0, min(1.0, v))

# Pre-computar matriz do treino
train['hora_n'] = train['hora'].apply(norm_hora)
train['acum_n'] = train['acumulado_ate_hora'].apply(norm_acum)
X_tr = train[['hora_n','acum_n']].values
y_tr = train['acordos_banda'].values.astype(float)

def testB_pred(hora, acum):
    q = np.array([norm_hora(hora), norm_acum(acum)])
    dists = np.sqrt(((X_tr - q)**2).sum(axis=1))
    if len(dists) < K:
        fallback_b[0] += 1
        return testA_pred(hora, acum)
    k_idx = np.argpartition(dists, K)[:K]
    return float(np.median(y_tr[k_idx]))

b_preds = np.array([testB_pred(r['hora'], r['acumulado_ate_hora'])
                    for _, r in holdout.iterrows()], dtype=float)
mask_b  = ~(np.isnan(b_preds) | np.isnan(real))
mae_b   = np.abs(b_preds[mask_b] - real[mask_b]).mean()

# ---- RESULTADOS ---------------------------------------------------------
print()
print('=== COMPARATIVO GLOBAL ===')
print('{:<16} {:>12} {:>10} {:>10} {:>10}'.format(
    'Metrica', 'Baseline', 'Test A', 'Test B', 'Delta A/B'))
print(SEP)
print('{:<16} {:>12.2f} {:>10.2f} {:>10.2f} {:>10}'.format(
    'MAE (acordos)', mae_base, mae_a, mae_b,
    '{:+.2f}/{:+.2f}'.format(mae_a - mae_base, mae_b - mae_base)))
print('{:<16} {:>12} {:>10} {:>10}'.format('N obs', int(mask_base.sum()), int(mask_a.sum()), int(mask_b.sum())))

print()
print('Test B -- OOD acum clipping: {}  |  Fallbacks: {}'.format(ood_b[0], fallback_b[0]))

# MAE por hora
print()
print('=== MAE POR HORA ===')
print('{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
    'Hora', 'Baseline', 'Test A', 'Delta A', 'Test B', 'Delta B'))
print(SEP)
for h in sorted(holdout['hora'].unique()):
    idx = holdout['hora'] == h
    rv  = real[idx.values]

    bp  = base_preds[idx.values]
    ap  = a_preds[idx.values]
    bp2 = b_preds[idx.values]

    m_base = np.abs(bp[~np.isnan(bp)] - rv[~np.isnan(bp)]).mean() if (~np.isnan(bp)).sum() else np.nan
    m_a    = np.abs(ap[~np.isnan(ap)] - rv[~np.isnan(ap)]).mean() if (~np.isnan(ap)).sum() else np.nan
    m_b    = np.abs(bp2[~np.isnan(bp2)] - rv[~np.isnan(bp2)]).mean() if (~np.isnan(bp2)).sum() else np.nan

    da = (m_a - m_base) if not np.isnan(m_a) else np.nan
    db = (m_b - m_base) if not np.isnan(m_b) else np.nan
    fa = ' <' if (not np.isnan(da) and da > 0.5) else ''
    fb = ' <' if (not np.isnan(db) and db > 0.5) else ''

    print('{:<6} {:>10.2f} {:>10.2f} {:>+10.2f}{} {:>10.2f} {:>+10.2f}{}'.format(
        h, m_base,
        m_a if not np.isnan(m_a) else 0, da if not np.isnan(da) else 0, fa,
        m_b if not np.isnan(m_b) else 0, db if not np.isnan(db) else 0, fb))

# MAE por faixa (Test A nao tem faixa -- mostrar baseline vs Test B por faixa)
print()
print('=== MAE POR FAIXA (baseline vs Test B) ===')
print('{:<16} {:>12} {:>10} {:>10}'.format('Faixa', 'Baseline', 'Test B', 'Delta B'))
print(SEP)
faixas_b_melhores = 0
for f in sorted(holdout['faixa_batimento'].unique()):
    idx = holdout['faixa_batimento'] == f
    rv  = real[idx.values]
    bp  = base_preds[idx.values]; mk_b = ~np.isnan(bp)
    bpb = b_preds[idx.values];    mk_b2 = ~np.isnan(bpb)
    m_base = np.abs(bp[mk_b] - rv[mk_b]).mean() if mk_b.sum() else np.nan
    m_b    = np.abs(bpb[mk_b2] - rv[mk_b2]).mean() if mk_b2.sum() else np.nan
    db = m_b - m_base if not (np.isnan(m_b) or np.isnan(m_base)) else np.nan
    if not np.isnan(db) and db < 0:
        faixas_b_melhores += 1
    print('{:<16} {:>12.2f} {:>10.2f} {:>+10.2f}  n={}'.format(
        f, m_base, m_b if not np.isnan(m_b) else 0,
        db if not np.isnan(db) else 0, idx.sum()))

# ---- DECISAO ------------------------------------------------------------
print()
print('=== DECISAO ===')
threshold   = 16.03 * 0.95   # melhora >= 5%
best_mae    = min(mae_a, mae_b)
best_label  = 'Test A' if mae_a <= mae_b else 'Test B'

print('Baseline MAE: {:.2f}  |  Threshold (5% melhora): {:.2f}'.format(mae_base, threshold))
print('Test A MAE:   {:.2f}  ({:+.1f}%)'.format(mae_a, (mae_a - mae_base)/mae_base*100))
print('Test B MAE:   {:.2f}  ({:+.1f}%)'.format(mae_b, (mae_b - mae_base)/mae_base*100))
print()

if best_mae < threshold:
    print('RESULTADO: PROMOVER {}  (MAE={:.2f}, delta={:+.2f})'.format(
        best_label, best_mae, best_mae - mae_base))
else:
    nenhum_melhora = mae_a >= mae_base and mae_b >= mae_base
    if nenhum_melhora:
        print('RESULTADO: MANTER 3 faixas -- faixa_batimento permanece relevante na janela abril.')
    else:
        print('RESULTADO: MANTER 3 faixas -- melhora existe mas abaixo do threshold de 5%.')
