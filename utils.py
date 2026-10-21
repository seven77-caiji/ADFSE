import os

# Windows 上 wmic 已被移除，joblib 探测物理核数会报 WinError 2 并打印整段 traceback。
# loky 只有在 LOKY_MAX_CPU_COUNT 严格小于逻辑核数时才跳过该探测，
# 所以这里设为逻辑核数 - 1（仅少一个核，对 n_jobs=-1 影响可忽略）。
_n_cpus = os.cpu_count() or 1
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(max(1, _n_cpus - 1)))

from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent


def get_device(name='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def load_data(config):
    data_dir = Path(config['data_dir'])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    mat = loadmat(data_dir / (config['dataset_name'] + '.mat'))
    if 'X' in mat and 'y' in mat:
        x, y = mat['X'], mat['y']
    else:
        x, y = mat['trandata'][:, :-1], mat['trandata'][:, -1]
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    indices = np.random.default_rng(config['seed']).permutation(len(y))
    normal = indices[y[indices] == 0]
    split = len(normal) // 2
    train = x[normal[:split]]
    test_indices = np.r_[normal[split:], indices[y[indices] != 0]]
    scaler = MinMaxScaler().fit(train)
    state = {name: getattr(scaler, name).astype(np.float32)
             for name in ('min_', 'scale_', 'data_min_', 'data_max_', 'data_range_')}
    return (scaler.transform(train).astype(np.float32),
            scaler.transform(x[test_indices]).astype(np.float32),
            (y[test_indices] != 0).astype(np.int64), state)


def load_config(path=None):
    with Path(path or ROOT / 'config.yml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    for key in ('datasets', 'seeds', 'c', 'model_config'):
        if key not in config:
            raise ValueError(f'Missing configuration key: {key}')
    if not config['datasets'] or len(set(config['datasets'])) != len(config['datasets']):
        raise ValueError('datasets must be a nonempty unique list')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must be a nonempty unique list')
    if any(not isinstance(s, int) or s < 0 for s in config['seeds']):
        raise ValueError('seeds must contain nonnegative integers')
    if not 2 <= config['c'] <= 20:
        raise ValueError('c must be between 2 and 20')
    return config


def parse_seeds(value):
    if '-' in value:
        first, last = map(int, value.split('-', 1))
        values = list(range(first, last + 1))
    else:
        values = [int(v.strip()) for v in value.split(',')]
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError('Seeds must be nonnegative and unique; e.g. 42-46 or 42,43')
    return values


def shared_settings(config):
    return {k: v for k, v in config['model_config'].items() if k not in ('dataset_name', 'seed')}


def run_settings(config, dataset, seed):
    return {**shared_settings(config), 'dataset_name': dataset, 'seed': seed,
            'k': config['c'], 'verbose': False}


def checkpoint_path(dataset, seed, weights_dir=None):
    return Path(weights_dir or ROOT / 'Weights') / dataset / f'seed_{seed}' / 'model.pth'
