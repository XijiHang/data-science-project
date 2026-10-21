"""PM2.5 forecast evaluation, adapted from my completed Practice 07 analysis.

No model choice, threshold, or hyperparameter is selected using test outcomes.
"""
from pathlib import Path
import os
import io
import json
import zipfile
import hashlib
import math

BASE = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = BASE
OUT.mkdir(exist_ok=True)
archive = BASE / 'data' / 'UCI501_Original.zip'
archive.parent.mkdir(exist_ok=True)
expected_sha256 = 'b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4'
if not archive.exists():
    from urllib.request import urlopen
    url = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
    print('Downloading the public UCI dataset...', flush=True)
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('The UCI archive has changed. Review its provenance before continuing.')
    archive.write_bytes(payload)
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
    raise ValueError('Dataset checksum mismatch.')

with zipfile.ZipFile(archive) as z:
    nested = [n for n in z.namelist() if n.endswith('.zip')]
    if nested:
        data_z = zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
    else:
        data_z = z
    names = sorted(n for n in data_z.namelist() if 'PRSA_Data_' in n and n.endswith('.csv'))
    assert len(names) == 12, names
    raw = pd.concat([pd.read_csv(data_z.open(n)) for n in names], ignore_index=True)

raw['timestamp'] = pd.to_datetime(raw[['year', 'month', 'day', 'hour']])
assert not raw.duplicated(['station', 'timestamp']).any()
assert len(raw) == 420768
assert raw['station'].nunique() == 12
assert (raw['PM2.5'].dropna() >= 0).all()
raw['date'] = raw['timestamp'].dt.floor('D')

NUMERIC = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
means = raw.groupby(['station', 'date'])[NUMERIC].mean()
feature_counts = raw.groupby(['station', 'date'])[NUMERIC].count()
means = means.mask(feature_counts < 18)
count = raw.groupby(['station', 'date'])['PM2.5'].count()
std = raw.groupby(['station', 'date'])['PM2.5'].std()
rain = raw.groupby(['station', 'date'])['RAIN'].sum(min_count=1)
rain = rain.mask(raw.groupby(['station', 'date'])['RAIN'].count() < 18)

stations = sorted(raw['station'].unique())
dates = pd.date_range('2013-03-01', '2017-02-28', freq='D')
idx = pd.MultiIndex.from_product([stations, dates], names=['station', 'date'])
daily = means.reindex(idx)
daily['pm25_count'] = count.reindex(idx)
daily['pm25_std'] = std.reindex(idx)
daily['rain_sum'] = rain.reindex(idx)

# Low-coverage day summaries are unavailable; do not fill targets or PM2.5.
low_coverage = daily['pm25_count'] < 18
daily.loc[low_coverage, ['PM2.5', 'pm25_std']] = np.nan
for lag in [1, 2, 7]:
    daily[f'pm25_lag{lag}'] = daily.groupby(level='station')['PM2.5'].shift(lag)
daily['pm25_mean7'] = daily.groupby(level='station')['PM2.5'].transform(
    lambda s: s.rolling(7, min_periods=5).mean())
daily['target'] = daily.groupby(level='station')['PM2.5'].shift(-1)
daily['target_count'] = daily.groupby(level='station')['pm25_count'].shift(-1)
daily = daily.reset_index()
daily['target_date'] = daily['date'] + pd.Timedelta(days=1)
doy = daily['target_date'].dt.dayofyear
daily['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
daily['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
daily['weekday'] = daily['target_date'].dt.dayofweek
daily['station'] = pd.Categorical(daily['station'], categories=stations)

eligible = daily['PM2.5'].notna() & daily['target'].notna() & (daily['target_count'] >= 18)
eligible &= daily['target_date'] <= pd.Timestamp('2017-02-28')
samples = daily.loc[eligible].copy()
cutoff = pd.Timestamp('2016-01-01')
train = samples[samples['target_date'] < cutoff].copy()
# Origin day must itself be on/after cutoff: remove the boundary crossing origin.
test = samples[samples['date'] >= cutoff].copy()
assert train['target_date'].max() < test['date'].min()
assert set(zip(train.station, train.date)).isdisjoint(set(zip(test.station, test.date)))
assert test['target_date'].min() == pd.Timestamp('2016-01-02')

FEATURES = NUMERIC + ['pm25_count', 'pm25_std', 'rain_sum', 'pm25_lag1', 'pm25_lag2',
                     'pm25_lag7', 'pm25_mean7', 'doy_sin', 'doy_cos', 'weekday', 'station']
X_train = train[FEATURES]
X_test = test[FEATURES]
y_train = train['target'].to_numpy()
y_test = test['target'].to_numpy()
thresholds = {f'q{q}': float(np.quantile(y_train, q / 100)) for q in [85, 90, 95]}
threshold = thresholds['q90']

params = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
              min_samples_leaf=30, l2_regularization=10,
              early_stopping=False, random_state=720407, categorical_features='from_dtype')
models = {}
preds = {'Persistence': test['PM2.5'].to_numpy()}
print(f'Raw hourly rows: {len(raw):,}; train station-days: {len(train):,}; test station-days: {len(test):,}', flush=True)
print(f'High-pollution cutoff, training P90: {threshold:.6f} ug/m3', flush=True)
for name, loss in [('Boosting MAE', 'absolute_error'), ('Boosting MSE', 'squared_error')]:
    model = HistGradientBoostingRegressor(loss=loss, **params)
    model.fit(X_train, y_train)
    preds[name] = np.clip(model.predict(X_test), 0, None)
    models[name] = model
    print(f'{name} fitted; test MAE={mean_absolute_error(y_test, preds[name]):.3f}', flush=True)

test_out = test[['station', 'date', 'target_date', 'PM2.5', 'pm25_count', 'target_count', 'target']].copy()
for name, pred in preds.items():
    test_out[name] = pred
test_out['station'] = test_out['station'].astype(str)
test_out.to_json(OUT / 'test_predictions.json', orient='records', date_format='iso', indent=1)

names_models = list(preds)
high = y_test >= threshold
cuts = [0, float(np.quantile(y_train, .5)), float(np.quantile(y_train, .75)), threshold, np.inf]
bins = np.digitize(y_test, cuts[1:-1], right=False)


def score_rows(indices):
    y = y_test[indices]
    h = y >= threshold
    b = bins[indices]
    values = []
    for name in names_models:
        p = preds[name][indices]
        err = p - y
        pred_h = p >= threshold
        tp = np.sum(h & pred_h)
        values.extend([
            np.mean(np.abs(err)), np.mean(np.abs(err[h])) if h.any() else np.nan,
            100 * tp / np.sum(h) if h.any() else np.nan,
            100 * tp / np.sum(pred_h) if pred_h.any() else np.nan,
            *[np.mean(err[b == k]) if (b == k).any() else np.nan for k in range(4)]
        ])
    return np.array(values)


# Resample calendar-day blocks, retaining every available station within each day.
# The same draws are applied to all models to preserve paired comparisons.
calendar = pd.date_range(test['target_date'].min(), test['target_date'].max(), freq='D')
by_date = {d: np.flatnonzero(test['target_date'].to_numpy() == np.datetime64(d)) for d in calendar}
day_rows = [by_date[d] for d in calendar]
N_DAYS = len(calendar)


def bootstrap(block_length, reps, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        starts = rng.integers(0, N_DAYS, size=math.ceil(N_DAYS / block_length))
        selected = ((starts[:, None] + np.arange(block_length)) % N_DAYS).ravel()[:N_DAYS]
        rows = np.concatenate([day_rows[i] for i in selected])
        values.append(score_rows(rows))
    return np.array(values)


point = score_rows(np.arange(len(test)))
boots = bootstrap(7, 2000, 720407)
ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)
metrics = {}
for i, name in enumerate(names_models):
    off = i * 8
    p = preds[name]
    pred_h = p >= threshold
    tp = int(np.sum(high & pred_h))
    fn = int(np.sum(high & ~pred_h))
    fp = int(np.sum(~high & pred_h))
    tn = int(np.sum(~high & ~pred_h))
    metrics[name] = {
        'mae_all': float(point[off]), 'mae_high': float(point[off + 1]),
        'recall_high_pct': float(point[off + 2]), 'precision_high_pct': float(point[off + 3]),
        'bias_by_observed_bin': point[off + 4:off + 8].tolist(),
        'mae_all_ci': ci[:, off].tolist(), 'mae_high_ci': ci[:, off + 1].tolist(),
        'recall_high_pct_ci': ci[:, off + 2].tolist(), 'precision_high_pct_ci': ci[:, off + 3].tolist(),
        'bias_by_observed_bin_ci': ci[:, off + 4:off + 8].T.tolist(),
        'rmse_all': float(np.sqrt(mean_squared_error(y_test, p))),
        'bias_high': float(np.mean(p[high] - y_test[high])),
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
    }

gain = 100 * (metrics['Persistence']['mae_all'] - metrics['Boosting MAE']['mae_all']) / metrics['Persistence']['mae_all']
delta_all = boots[:, 8] - boots[:, 0]
delta_high = boots[:, 9] - boots[:, 1]
delta_recall = boots[:, 10] - boots[:, 2]
paired = {
    'Boosting MAE minus Persistence': {
        'mae_all_difference': float(point[8] - point[0]),
        'mae_all_difference_ci': np.percentile(delta_all, [2.5, 97.5]).tolist(),
        'mae_high_difference': float(point[9] - point[1]),
        'mae_high_difference_ci': np.percentile(delta_high, [2.5, 97.5]).tolist(),
        'recall_difference_pct_points': float(point[10] - point[2]),
        'recall_difference_pct_points_ci': np.percentile(delta_recall, [2.5, 97.5]).tolist(),
    }
}

sens = {'threshold': {}, 'block_length': {}, 'coverage': {}}
for key, cut in thresholds.items():
    h = y_test >= cut
    sens['threshold'][key] = {'cutoff': cut, 'n_high': int(h.sum()), 'models': {}}
    for name, p in preds.items():
        ph = p >= cut
        tp = np.sum(h & ph)
        sens['threshold'][key]['models'][name] = {
            'mae_high': float(np.mean(np.abs(p[h] - y_test[h]))),
            'recall_pct': float(100 * tp / h.sum()),
            'precision_pct': float(100 * tp / ph.sum()) if ph.any() else None,
        }
for block in [3, 14]:
    b = bootstrap(block, 1000, 720407 + block)
    sens['block_length'][str(block)] = {
        'reps': len(b), 'mae_difference_ci': np.percentile(b[:, 8] - b[:, 0], [2.5, 97.5]).tolist(),
        'high_mae_difference_ci': np.percentile(b[:, 9] - b[:, 1], [2.5, 97.5]).tolist(),
        'recall_difference_ci': np.percentile(b[:, 10] - b[:, 2], [2.5, 97.5]).tolist(),
    }
for coverage in [20, 24]:
    rows = np.flatnonzero((test['pm25_count'] >= coverage) & (test['target_count'] >= coverage))
    s = score_rows(rows)
    sens['coverage'][str(coverage)] = {'n_rows': len(rows), 'n_high': int((y_test[rows] >= threshold).sum()),
                                     'scores': {n: {'mae_all': float(s[i * 8]), 'mae_high': float(s[i * 8 + 1]),
                                                     'recall_pct': float(s[i * 8 + 2])} for i, n in enumerate(names_models)}}

results = {
    'author': 'Xiji Hang',
    'source': {'url': 'https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data',
               'download_url': 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip',
               'doi': '10.24432/C5RK5G', 'license': 'CC BY 4.0', 'access_date': '2026-10-07',
               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
    'raw_rows': len(raw), 'stations': stations, 'raw_pm25_missing': int(raw['PM2.5'].isna().sum()),
    'daily_grid_rows': len(daily), 'eligible_rows': len(samples), 'training_rows': len(train), 'test_rows': len(test),
    'omitted_boundary_rows': len(samples) - len(train) - len(test),
    'training_origin_range': [str(train.date.min().date()), str(train.date.max().date())],
    'training_target_range': [str(train.target_date.min().date()), str(train.target_date.max().date())],
    'test_origin_range': [str(test.date.min().date()), str(test.date.max().date())],
    'test_target_range': [str(test.target_date.min().date()), str(test.target_date.max().date())],
    'test_calendar_days': N_DAYS, 'test_high_calendar_days': int(test.loc[high, 'target_date'].nunique()),
    'high_threshold': threshold, 'test_high_rows': int(high.sum()), 'test_high_percent': 100 * float(high.mean()),
    'observed_bin_cutpoints': cuts[:-1], 'observed_bin_counts': [int(np.sum(bins == k)) for k in range(4)],
    'models': metrics, 'paired_comparisons': paired, 'mae_improvement_percent': gain,
    'model_parameters': params, 'feature_columns': FEATURES,
    'bootstrap': {'type': 'circular calendar-day block bootstrap', 'block_days': 7, 'replicates': 2000,
                  'seed': 720407, 'ci': '95% percentile', 'all_stations_resampled_together': True,
                  'refit_models': False},
    'sensitivity': sens, 'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
}
(OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps({k: results[k] for k in ['training_rows', 'test_rows', 'high_threshold', 'test_high_rows',
                                         'test_high_percent', 'models', 'paired_comparisons', 'mae_improvement_percent']}, indent=2), flush=True)

from plot_results import make_figures
make_figures()
