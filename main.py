"""
CLI entry point for the HSI Computational Intelligence experiment runner.

Usage examples:
    python main.py --help
    python main.py --dataset all --data-dir ./data/raw --results-dir ./results
    python main.py --dataset indian_pines --experiment deep_learning
"""

import argparse
import sys

from main_experiment import ExperimentRunner

DATASETS = ['indian_pines', 'pavia_university', 'salinas_valley', 'houston', 'botswana', 'all']
EXPERIMENTS = ['deep_learning', 'watershed', 'thresholding', 'all']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'HSI Computational Intelligence — Thesis Experiment Runner\n'
            'Optimisation of Computational Intelligence Techniques for '
            'Hyperspectral Image Segmentation'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        default='all',
        choices=DATASETS,
        help='Dataset to run experiments on (default: all)',
    )
    parser.add_argument(
        '--data-dir',
        default='./data/raw',
        metavar='PATH',
        help='Directory containing .mat dataset files (default: ./data/raw)',
    )
    parser.add_argument(
        '--results-dir',
        default='./results',
        metavar='PATH',
        help='Directory to write result files (default: ./results)',
    )
    parser.add_argument(
        '--experiment',
        default='all',
        choices=EXPERIMENTS,
        help='Experiment type to run (default: all)',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import os
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    runner = ExperimentRunner(data_dir=args.data_dir, results_dir=args.results_dir)

    datasets = (
        ['indian_pines', 'pavia_university', 'salinas_valley', 'houston', 'botswana']
        if args.dataset == 'all'
        else [args.dataset]
    )

    import numpy as np

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name.upper()}")
        print(f"{'='*60}")

        data, labels = runner.data_loader.load_data(dataset_name)
        if data is None:
            print(f"Skipping {dataset_name}: Data not found")
            continue

        split_info = runner.data_loader.split_data(labels, train_ratio=0.7)
        train_mask = split_info['train_mask']
        test_mask = split_info['test_mask']

        if args.experiment in ('deep_learning', 'all'):
            runner._run_deep_learning_experiment(
                dataset_name, data, labels, train_mask, test_mask, split_info
            )
        if args.experiment in ('watershed', 'all'):
            runner._run_watershed_experiment(
                dataset_name, data, labels, train_mask, test_mask, split_info
            )
        if args.experiment in ('thresholding', 'all'):
            runner._run_thresholding_experiment(
                dataset_name, data, labels, train_mask, test_mask, split_info
            )

    output_file = runner.save_results()
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

