"""Reproduce ADFSE: load saved weights and print AUROC / AUCPR mean ± std per dataset."""

import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import numpy as np

from Model.Trainer import Trainer
from utils import ROOT, checkpoint_path, load_config, parse_seeds, run_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yml')
    parser.add_argument('--dataset', nargs='+', help='Default: all configured datasets')
    parser.add_argument('--seeds', help='Default: five configured seeds; e.g. 42-46 or 42,43')
    args = parser.parse_args()
    config = load_config(args.config)
    datasets = args.dataset or config['datasets']
    seeds = parse_seeds(args.seeds) if args.seeds else config['seeds']
    if len(set(datasets)) != len(datasets) or any(d not in config['datasets'] for d in datasets):
        parser.error('Dataset names must be unique and listed in config.yml')

    print(f'Reproduce ADFSE: {len(datasets)} datasets x {len(seeds)} seeds; loads saved weights only', flush=True)
    per_dataset = {}
    for dataset in datasets:
        aurocs, auprs = [], []
        for seed in seeds:
            path = checkpoint_path(dataset, seed)
            if not path.exists():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
            trainer = Trainer({**run_settings(config, dataset, seed), 'load_only': True})
            trainer.load_model(path)
            auroc, aucpr = trainer.evaluate()
            aurocs.append(auroc)
            auprs.append(aucpr)
            del trainer
        aurocs = np.asarray(aurocs) * 100
        auprs = np.asarray(auprs) * 100
        per_dataset[dataset] = (aurocs, auprs)
        print(f'{dataset:20s} AUROC={aurocs.mean():.2f}+/-{aurocs.std(ddof=0):.2f}%  '
              f'AUCPR={auprs.mean():.2f}+/-{auprs.std(ddof=0):.2f}%', flush=True)

    macro_auroc = np.mean([per_dataset[d][0].mean() for d in datasets])
    macro_aupr = np.mean([per_dataset[d][1].mean() for d in datasets])
    print(f'Macro-average over {len(datasets)} datasets: AUROC={macro_auroc:.2f}%  AUCPR={macro_aupr:.2f}%', flush=True)


if __name__ == '__main__':
    main()
