"""Train ADFSE from scratch with the saved common hyperparameters and five seeds."""

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

    print(f'Fresh training: {len(datasets)} datasets x {len(seeds)} seeds, c={config["c"]}', flush=True)
    aurocs, auprs = [], []
    for dataset in datasets:
        for seed in seeds:
            print(f'Training {dataset} seed={seed} c={config["c"]}', flush=True)
            trainer = Trainer(run_settings(config, dataset, seed))
            trainer.training()
            trainer.save_model(checkpoint_path(dataset, seed))
            auroc, aucpr = trainer.evaluate()
            aurocs.append(auroc)
            auprs.append(aucpr)
            print(f'  AUROC={100*auroc:.2f}%  AUCPR={100*aucpr:.2f}%', flush=True)
            del trainer
    print(f'Overall AUROC={100*np.mean(aurocs):.2f}%  AUCPR={100*np.mean(auprs):.2f}%', flush=True)


if __name__ == '__main__':
    main()
