"""
KNN Phase 2 — Ritmo do Dia
Usa sklearn: StandardScaler, KNeighborsRegressor, train_test_split, MAE.
Target: acordos_banda (numérico) → regressão, não classificação.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── STEP 1: DATA PREPARATION ──────────────────────────────────────────────
print('=' * 62)
print('STEP 1 — DATA PREPARATION')
print('=' * 62)

COLS = ['banco_origem','dia','hora','dia_semana','total_dia',
        'acumulado_ate_hora','proporcao_ate_hora',
        'data_ultimo_batimento','dias_desde_ultimo_batimento','faixa_batimento']

df = pd.read_csv('d.csv', sep=';', decimal=',', names=COLS)
df['dia'] = pd.to_datetime(df['dia'])
df['acumulado_ate_hora'] = pd.to_numeric(df['acumulado_ate_hora'], errors='coerce')
df['dias_desde_ultimo_batimento'] = pd.to_numeric(
    df['dias_desde_ultimo_batimento'], errors='coerce')

# Imputar NULL dias_desde_ultimo_batimento (basais antigos sem batimento)
n_null = df['dias_desde_ultimo_batimento'].isna().sum()
df['dias_desde_ultimo_batimento'] = df['dias_desde_ultimo_batimento'].fillna(30)
print('NULLs imputados (dias_desde_ultimo_batimento=30): {}'.format(n_null))

# Derivar acordos_banda = incremental por hora dentro do dia
df = df.sort_values(['banco_origem','dia','hora']).reset_index(drop=True)
df['acordos_banda'] = (df.groupby(['banco_origem','dia'])['acumulado_ate_hora']
                         .diff()
                         .fillna(df['acumulado_ate_hora'])
                         .astype(int))
df['acordos_banda'] = df['acordos_banda'].clip(lower=0)

print('Shape: {}  |  dias: {}  |  faixas: {}'.format(
    df.shape, df['dia'].nunique(), sorted(df['faixa_batimento'].unique())))
print('acordos_banda — min:{} max:{} median:{:.1f}'.format(
    df['acordos_banda'].min(), df['acordos_banda'].max(), df['acordos_banda'].median()))

# ── STEP 2: SPLIT TEMPORAL ────────────────────────────────────────────────
print()
print('=' * 62)
print('STEP 2 — SPLIT TEMPORAL (110 treino / 20 holdout)')
print('=' * 62)

dias_ord   = sorted(df['dia'].unique())
TRAIN_DAYS = 110
train_dias   = set(dias_ord[:TRAIN_DAYS])
holdout_dias = set(dias_ord[-20:])
assert not train_dias & holdout_dias

train   = df[df['dia'].isin(train_dias)].copy().reset_index(drop=True)
holdout = df[df['dia'].isin(holdout_dias)].copy().reset_index(drop=True)

print('Treino:  {} dias ({} linhas)  {} -> {}'.format(
    len(train_dias), len(train),
    pd.Timestamp(min(train_dias)).date(), pd.Timestamp(max(train_dias)).date()))
print('Holdout: {} dias ({} linhas)  {} -> {}'.format(
    len(holdout_dias), len(holdout),
    pd.Timestamp(min(holdout_dias)).date(), pd.Timestamp(max(holdout_dias)).date()))

# Phase 1 lookup (mediana por hora+faixa) — baseline e fallback
phase1_lookup = (train.groupby(['hora','faixa_batimento'])['acordos_banda']
                      .median().round().astype(int).to_dict())

# ── STEP 3: K SELECTION (walk-forward CV no treino) ──────────────────────
print()
print('=' * 62)
print('STEP 3 — K SELECTION (walk-forward CV, 4 folds)')
print('=' * 62)

FEATURES   = ['dias_desde_ultimo_batimento', 'acumulado_ate_hora']
K_OPTS     = [3, 5, 7, 10]
N_FOLDS    = 4
FOLD_SIZE  = 10
FOLD_BASE  = TRAIN_DAYS - N_FOLDS * FOLD_SIZE  # 70

train_dias_list = sorted(train_dias)
cv_mae = {k: [] for k in K_OPTS}

for fold in range(N_FOLDS):
    t_start = FOLD_BASE + fold * FOLD_SIZE
    t_end   = t_start + FOLD_SIZE
    fd_train = train[train['dia'].isin(set(train_dias_list[:t_start]))].copy()
    fd_test  = train[train['dia'].isin(set(train_dias_list[t_start:t_end]))].copy()

    # StandardScaler ajustado somente no fold_train
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(fd_train[FEATURES])
    X_te = scaler.transform(fd_test[FEATURES])
    y_tr = fd_train['acordos_banda'].values
    y_te = fd_test['acordos_banda'].values

    for k in K_OPTS:
        knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
        knn.fit(X_tr, y_tr)
        preds = knn.predict(X_te)
        mae   = mean_absolute_error(y_te, preds)
        cv_mae[k].append(mae)

print('{:<6} {:>8} {:>8} {:>8} {:>8} {:>10}'.format(
    'k', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'MAE medio'))
print('-' * 52)
best_k, best_cv_mae = None, float('inf')
for k in K_OPTS:
    vals     = cv_mae[k]
    mean_mae = np.mean(vals)
    print('{:<6} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>10.2f}'.format(
        k, *vals, mean_mae))
    if mean_mae < best_cv_mae:
        best_cv_mae = mean_mae
        best_k = k

if best_k < 5:
    print('k={} selecionado mas k<5 proibido. Usando k=5.'.format(best_k))
    best_k = 5

print()
print('k selecionado: {}  (MAE CV: {:.2f})'.format(best_k, best_cv_mae))

# ── STEP 4: HOLDOUT EVALUATION ────────────────────────────────────────────
print()
print('=' * 62)
print('STEP 4 — HOLDOUT EVALUATION  (k={})'.format(best_k))
print('=' * 62)

# Scaler ajustado no treino completo (110 dias)
scaler_final = StandardScaler()
X_train_full = scaler_final.fit_transform(train[FEATURES])
y_train_full = train['acordos_banda'].values

knn_final = KNeighborsRegressor(n_neighbors=best_k, metric='euclidean')
knn_final.fit(X_train_full, y_train_full)

X_hold = scaler_final.transform(holdout[FEATURES])
y_hold = holdout['acordos_banda'].values

# KNN: predição global (sklearn) — sem fallback necessário pois treino tem todos os grupos
knn_preds_raw = knn_final.predict(X_hold)
knn_preds     = np.round(knn_preds_raw).astype(int)

# Phase 1: mediana por (hora, faixa)
p1_preds = np.array([
    phase1_lookup.get((r['hora'], r['faixa_batimento']), np.nan)
    for _, r in holdout.iterrows()], dtype=float)

mae_knn = mean_absolute_error(y_hold, knn_preds)
mae_p1  = mean_absolute_error(y_hold[~np.isnan(p1_preds)],
                               p1_preds[~np.isnan(p1_preds)])

# OOD check: valores fora do range de treino
ood_dias = ((holdout['dias_desde_ultimo_batimento'] < train['dias_desde_ultimo_batimento'].min()) |
            (holdout['dias_desde_ultimo_batimento'] > train['dias_desde_ultimo_batimento'].max())).sum()
ood_acum = ((holdout['acumulado_ate_hora'] < train['acumulado_ate_hora'].min()) |
            (holdout['acumulado_ate_hora'] > train['acumulado_ate_hora'].max())).sum()
print('OOD dias_desde_batimento: {}  |  OOD acumulado_ate_hora: {}'.format(ood_dias, ood_acum))
print()

print('{:<22} {:>10} {:>10} {:>10}'.format('Metrica', 'Phase 1', 'KNN k={}'.format(best_k), 'Delta'))
print('-' * 54)
print('{:<22} {:>10.2f} {:>10.2f} {:>+10.2f}'.format('MAE (acordos)', mae_p1, mae_knn, mae_knn - mae_p1))
print('{:<22} {:>10}'.format('N obs holdout', len(y_hold)))

# Estratificação por faixa
print()
print('{:<22} {:>10} {:>10} {:>10}'.format('Faixa', 'Phase 1', 'KNN', 'Delta'))
print('-' * 54)
holdout['knn_pred'] = knn_preds
holdout['p1_pred']  = p1_preds
for f in sorted(holdout['faixa_batimento'].unique()):
    sub = holdout[holdout['faixa_batimento'] == f]
    m_p1  = mean_absolute_error(sub['acordos_banda'], sub['p1_pred'].fillna(sub['acordos_banda']))
    m_knn = mean_absolute_error(sub['acordos_banda'], sub['knn_pred'])
    print('{:<22} {:>10.2f} {:>10.2f} {:>+10.2f}   n={}'.format(
        f, m_p1, m_knn, m_knn - m_p1, len(sub)))

# Estratificação por hora
print()
print('{:<8} {:>10} {:>10} {:>10}'.format('Hora', 'Phase 1', 'KNN', 'Delta'))
print('-' * 40)
for h in sorted(holdout['hora'].unique()):
    sub = holdout[holdout['hora'] == h]
    m_p1  = mean_absolute_error(sub['acordos_banda'], sub['p1_pred'].fillna(sub['acordos_banda']))
    m_knn = mean_absolute_error(sub['acordos_banda'], sub['knn_pred'])
    print('{:<8} {:>10.2f} {:>10.2f} {:>+10.2f}   n={}'.format(
        h, m_p1, m_knn, m_knn - m_p1, len(sub)))

# ── DECISAO ───────────────────────────────────────────────────────────────
print()
print('=' * 62)
print('DECISAO DE PROMOCAO')
print('=' * 62)

delta     = mae_knn - mae_p1
delta_pct = delta / mae_p1 * 100
promover  = delta < -0.5

print('MAE Phase 1 (baseline): {:.2f}'.format(mae_p1))
print('MAE KNN k={}:           {:.2f}'.format(best_k, mae_knn))
print('Delta:                  {:+.2f} ({:+.1f}%)'.format(delta, delta_pct))
print()
if promover:
    print('RESULTADO: PROMOVER KNN — redução de {:.2f} acordos/banda ({:.1f}%)'.format(
        abs(delta), abs(delta_pct)))
else:
    print('RESULTADO: MANTER Phase 1 — KNN não reduziu MAE em >= 0.5 acordos.')
    print('           Complexidade não justificada (delta={:+.2f}).'.format(delta))

print()
print('Parâmetros para Step 5 (integração FastAPI):')
print('  best_k    = {}'.format(best_k))
print('  scaler    = StandardScaler (mean={}, std={})'.format(
    scaler_final.mean_.tolist(), scaler_final.scale_.tolist()))
