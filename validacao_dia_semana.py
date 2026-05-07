import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('d.csv', sep=';', decimal=',',
                 names=['banco_origem','dia','hora','dia_semana','total_dia',
                        'acumulado_ate_hora','proporcao_ate_hora',
                        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento'])

df['dia'] = pd.to_datetime(df['dia'])
df['proporcao_ate_hora'] = pd.to_numeric(df['proporcao_ate_hora'], errors='coerce')
df['dia_semana'] = pd.to_numeric(df['dia_semana'], errors='coerce')

# ── PASSO 1: ESPARSIDADE ───────────────────────────────────────────────────
print('=' * 60)
print('PASSO 1 — ESPARSIDADE (hora x faixa_batimento x dia_semana)')
print('=' * 60)

contagens = (df.groupby(['hora','faixa_batimento','dia_semana'])
               .size()
               .reset_index(name='n'))

total_celulas     = len(contagens)
celulas_possiveis = 12 * 3 * 5
celulas_ausentes  = celulas_possiveis - total_celulas
celulas_lt5       = (contagens['n'] < 5).sum()
celulas_lt10      = (contagens['n'] < 10).sum()

desc = contagens['n'].describe(percentiles=[.25,.5,.75])
print('Resumo de observacoes por celula:')
print('  Min:    {}'.format(int(desc['min'])))
print('  Q1:     {:.1f}'.format(desc['25%']))
print('  Mediana:{:.1f}'.format(desc['50%']))
print('  Q3:     {:.1f}'.format(desc['75%']))
print('  Max:    {}'.format(int(desc['max'])))
print('  Media:  {:.1f}'.format(desc['mean']))
print()
print('Celulas possiveis (12h x 3 faixas x 5 dias): {}'.format(celulas_possiveis))
print('Celulas com dados:   {}'.format(total_celulas))
print('Celulas ausentes:    {}'.format(celulas_ausentes))
print('Celulas com n < 5:   {}'.format(celulas_lt5))
print('Celulas com n < 10:  {}'.format(celulas_lt10))

if celulas_lt5 > 0:
    print()
    print('Celulas com n < 5:')
    for _, row in contagens[contagens['n'] < 5].iterrows():
        print('  hora={} faixa={} dia_semana={}  n={}'.format(
            int(row['hora']), row['faixa_batimento'], int(row['dia_semana']), int(row['n'])))

# ── PASSO 2: LOOKUPS ───────────────────────────────────────────────────────
print()
print('=' * 60)
print('PASSO 2 — CONSTRUCAO DOS LOOKUPS')
print('=' * 60)

dias_ordenados = sorted(df['dia'].unique())
holdout_dias   = set(dias_ordenados[-30:])
treino         = df[~df['dia'].isin(holdout_dias)].copy()
teste          = df[df['dia'].isin(holdout_dias)].copy()

print('Treino: {} dias ({} linhas)'.format(treino['dia'].nunique(), len(treino)))
print('Teste:  {} dias ({} linhas)'.format(teste['dia'].nunique(), len(teste)))
print('Holdout: {} -> {}'.format(
    pd.Timestamp(min(holdout_dias)).date(),
    pd.Timestamp(max(holdout_dias)).date()))

lookup_2d = (treino
    .groupby(['hora','faixa_batimento'])['proporcao_ate_hora']
    .median()
    .to_dict())

lookup_3d = (treino
    .groupby(['hora','faixa_batimento','dia_semana'])['proporcao_ate_hora']
    .median()
    .to_dict())

fallbacks_usados = [0]

def proj_2d(row):
    prop = lookup_2d.get((row['hora'], row['faixa_batimento']))
    if prop and prop > 0:
        return row['acumulado_ate_hora'] / prop
    return np.nan

def proj_3d(row):
    prop = lookup_3d.get((row['hora'], row['faixa_batimento'], row['dia_semana']))
    if prop and prop > 0:
        return row['acumulado_ate_hora'] / prop
    # fallback para 2D
    prop = lookup_2d.get((row['hora'], row['faixa_batimento']))
    if prop and prop > 0:
        fallbacks_usados[0] += 1
        return row['acumulado_ate_hora'] / prop
    return np.nan

# ── PASSO 3: AVALIACAO ─────────────────────────────────────────────────────
print()
print('=' * 60)
print('PASSO 3 — AVALIACAO NO HOLDOUT')
print('=' * 60)

teste = teste[teste['proporcao_ate_hora'] > 0].copy()
teste['p2d'] = teste.apply(proj_2d, axis=1)
teste['p3d'] = teste.apply(proj_3d, axis=1)

print('Fallbacks 3D->2D usados: {}/{}'.format(fallbacks_usados[0], len(teste)))

valid = teste.dropna(subset=['p2d','p3d']).copy()
valid['real'] = valid['total_dia']

valid['e2d'] = (valid['p2d'] - valid['real']).abs()
valid['e3d'] = (valid['p3d'] - valid['real']).abs()
valid['r2d'] = valid['e2d'] / valid['real'].clip(lower=1)
valid['r3d'] = valid['e3d'] / valid['real'].clip(lower=1)

mae_2d  = valid['e2d'].mean()
mae_3d  = valid['e3d'].mean()
mape_2d = valid['r2d'].mean() * 100
mape_3d = valid['r3d'].mean() * 100

t_stat, p_val = stats.ttest_rel(valid['e2d'], valid['e3d'])

print()
print('{:<22} {:>10} {:>10} {:>10}'.format('Metrica', '2D (base)', '3D (+dia)', 'Delta'))
print('-' * 54)
print('{:<22} {:>10.2f} {:>10.2f} {:>+10.2f}'.format('MAE (acordos)', mae_2d, mae_3d, mae_3d - mae_2d))
print('{:<22} {:>10.1f} {:>10.1f} {:>+10.1f}'.format('MAPE (%)', mape_2d, mape_3d, mape_3d - mape_2d))
print('{:<22} {:>10}'.format('N obs holdout', len(valid)))
print()
print('Teste t pareado: t={:.3f}, p={:.4f}'.format(t_stat, p_val))
sig = 'SIGNIFICATIVA (p<0.05)' if p_val < 0.05 else 'NAO significativa (p>=0.05)'
print('Diferenca: {}'.format(sig))

# MAE por faixa
print()
print('{:<22} {:>10} {:>10} {:>10}'.format('Faixa', 'MAE 2D', 'MAE 3D', 'Delta'))
print('-' * 54)
for f in sorted(valid['faixa_batimento'].unique()):
    sub = valid[valid['faixa_batimento'] == f]
    m2  = sub['e2d'].mean()
    m3  = sub['e3d'].mean()
    print('{:<22} {:>10.2f} {:>10.2f} {:>+10.2f}   n={}'.format(f, m2, m3, m3 - m2, len(sub)))

# MAE por hora
print()
print('{:<10} {:>10} {:>10} {:>10}'.format('Hora', 'MAE 2D', 'MAE 3D', 'Delta'))
print('-' * 42)
for h in sorted(valid['hora'].unique()):
    sub = valid[valid['hora'] == h]
    m2  = sub['e2d'].mean()
    m3  = sub['e3d'].mean()
    print('{:<10} {:>10.2f} {:>10.2f} {:>+10.2f}   n={}'.format(h, m2, m3, m3 - m2, len(sub)))

# ── PASSO 4: DECISAO ───────────────────────────────────────────────────────
print()
print('=' * 60)
print('PASSO 4 — DECISAO')
print('=' * 60)

delta_pct    = (mae_3d - mae_2d) / mae_2d * 100
melhora_5pct = delta_pct <= -5
esparso_ok   = celulas_lt5 <= (celulas_possiveis * 0.10)  # <10% celulas criticas

print('Reducao MAE >= 5%: {} ({:+.1f}%)'.format(melhora_5pct, delta_pct))
print('Esparsidade OK (<10% celulas criticas): {} ({} celulas <5 obs)'.format(esparso_ok, celulas_lt5))

if melhora_5pct and esparso_ok:
    print('RECOMENDACAO: ADOTAR modelo 3D (hora x faixa x dia_semana).')
elif melhora_5pct and not esparso_ok:
    print('RECOMENDACAO: MANTER 2D — ganho existe mas esparsidade e critica.')
else:
    print('RECOMENDACAO: MANTER 2D — ganho insuficiente (<5%) para justificar complexidade.')
