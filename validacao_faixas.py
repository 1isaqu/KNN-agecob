import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('d.csv', sep=';', decimal=',',
                 names=['banco_origem','dia','hora','dia_semana','total_dia',
                        'acumulado_ate_hora','proporcao_ate_hora',
                        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento'])

df['dia'] = pd.to_datetime(df['dia'])
df['proporcao_ate_hora'] = pd.to_numeric(df['proporcao_ate_hora'], errors='coerce')

dias_ordenados = sorted(df['dia'].unique())
holdout_dias = set(dias_ordenados[-30:])
treino = df[~df['dia'].isin(holdout_dias)].copy()
teste  = df[df['dia'].isin(holdout_dias)].copy()

print('Treino: {} dias | Teste: {} dias'.format(treino['dia'].nunique(), teste['dia'].nunique()))
print('Treino linhas: {} | Teste linhas: {}'.format(len(treino), len(teste)))
print('Holdout: {} -> {}'.format(pd.Timestamp(min(holdout_dias)).date(), pd.Timestamp(max(holdout_dias)).date()))

# Lookup 3-faixas
lookup_3f = (treino
    .groupby(['hora','faixa_batimento'])['proporcao_ate_hora']
    .median()
    .to_dict())

# Lookup 2-faixas
treino['faixa_simpl'] = treino['faixa_batimento'].apply(
    lambda x: 'pos_reshuffle' if x in ('pos_batimento','absorcao') else 'basal')
lookup_2f = (treino
    .groupby(['hora','faixa_simpl'])['proporcao_ate_hora']
    .median()
    .to_dict())

# Holdout: filtrar proporcao > 0 e adicionar faixa simplificada
teste = teste[teste['proporcao_ate_hora'] > 0].copy()
teste['faixa_simpl'] = teste['faixa_batimento'].apply(
    lambda x: 'pos_reshuffle' if x in ('pos_batimento','absorcao') else 'basal')

def proj(row, lookup, faixa_col):
    prop = lookup.get((row['hora'], row[faixa_col]))
    if prop and prop > 0:
        return row['acumulado_ate_hora'] / prop
    return np.nan

teste['proj_3f'] = teste.apply(lambda r: proj(r, lookup_3f, 'faixa_batimento'), axis=1)
teste['proj_2f'] = teste.apply(lambda r: proj(r, lookup_2f, 'faixa_simpl'), axis=1)

valid = teste.dropna(subset=['proj_3f','proj_2f']).copy()
valid['real'] = valid['total_dia']

valid['err_abs_3f'] = (valid['proj_3f'] - valid['real']).abs()
valid['err_abs_2f'] = (valid['proj_2f'] - valid['real']).abs()
valid['err_rel_3f'] = valid['err_abs_3f'] / valid['real'].clip(lower=1)
valid['err_rel_2f'] = valid['err_abs_2f'] / valid['real'].clip(lower=1)

mae_3f  = valid['err_abs_3f'].mean()
mae_2f  = valid['err_abs_2f'].mean()
mape_3f = valid['err_rel_3f'].mean() * 100
mape_2f = valid['err_rel_2f'].mean() * 100

t_stat, p_val = stats.ttest_rel(valid['err_abs_3f'], valid['err_abs_2f'])

print()
print('=== COMPARATIVO ===')
print('{:<20} {:>12} {:>12} {:>10}'.format('Metrica', '3-faixas', '2-faixas', 'Delta'))
print('-' * 56)
print('{:<20} {:>12.2f} {:>12.2f} {:>+10.2f}'.format('MAE (acordos)', mae_3f, mae_2f, mae_2f - mae_3f))
print('{:<20} {:>12.1f} {:>12.1f} {:>+10.1f}'.format('MAPE (%)', mape_3f, mape_2f, mape_2f - mape_3f))
print('{:<20} {:>12}'.format('N observacoes', len(valid)))
print()
print('Teste t pareado: t={:.3f}, p={:.4f}'.format(t_stat, p_val))
sig = 'SIGNIFICATIVA (p<0.05)' if p_val < 0.05 else 'NAO significativa (p>=0.05)'
print('Diferenca estatistica: {}'.format(sig))
print()

print('=== MAE POR FAIXA (3-faixas, no holdout) ===')
for f in sorted(valid['faixa_batimento'].unique()):
    sub = valid[valid['faixa_batimento'] == f]
    print('  {:<22}: MAE={:.2f}  MAPE={:.1f}%  n={}'.format(
        f, sub['err_abs_3f'].mean(), sub['err_rel_3f'].mean() * 100, len(sub)))

print()
print('=== MAE POR FAIXA (2-faixas, no holdout) ===')
for f in sorted(valid['faixa_simpl'].unique()):
    sub = valid[valid['faixa_simpl'] == f]
    print('  {:<22}: MAE={:.2f}  MAPE={:.1f}%  n={}'.format(
        f, sub['err_abs_2f'].mean(), sub['err_rel_2f'].mean() * 100, len(sub)))

print()
print('=== DECISAO ===')
delta_mae_pct = (mae_2f - mae_3f) / mae_3f * 100
ok_threshold  = mae_2f < 10 and mape_2f < 20
ok_degradacao = delta_mae_pct <= 5
print('2-faixas dentro dos thresholds (MAE<10, MAPE<20%): {}'.format(ok_threshold))
print('Degradacao MAE <= 5%: {} ({:+.1f}%)'.format(ok_degradacao, delta_mae_pct))
if ok_threshold and ok_degradacao:
    print('RECOMENDACAO: SIMPLIFICAR para 2 faixas.')
elif ok_threshold:
    print('RECOMENDACAO: MANTER 3 faixas — degradacao acima de 5%.')
else:
    print('RECOMENDACAO: MANTER 3 faixas — fora dos thresholds.')
