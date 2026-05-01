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

    print("="*60)
    print("HSI Computational Intelligence - Thesis Experiments")
    print("="*60)
    print("\nRunning experiments as described in the thesis:")
    print("- Deep Learning Enhancement (ILS-HED)")
    print("- Watershed Enhancement (WS-GA)")
    print("- Thresholding Enhancement (DE-GA)")
    print("- Evaluation on 5 benchmark datasets")
    print("- 70/30 stratified split protocol")
    print("- Statistical significance testing")
    print("\nNote: This code reproduces the methodology documented in the thesis")
    print("For full replication, download the dataset files from:")
    print("- Indian Pines, Pavia University, Salinas Valley")
    print("- Houston, Botswana")
    print("="*60)

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

    print("\n" + "="*60)
    print("Experiments complete!")
    print(f"Results saved to: {output_file}")
    print("="*60)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS (OA%)")
    print("="*60)
    print(f"{'Dataset':<20} {'Baseline HED':<12} {'ILS-HED':<12} {'Baseline WS':<12} {'WS-GA':<12} {'Baseline BT':<12} {'DE-GA':<12}")
    print("-"*60)

    for dataset in ['indian_pines', 'pavia_university', 'salinas_valley', 'houston', 'botswana']:
        dl_key = f"{dataset}_deep_learning"
        ws_key = f"{dataset}_watershed"
        th_key = f"{dataset}_thresholding"

        if dl_key in runner.results:
            dl_baseline = runner.results[dl_key]['baseline']['overall_accuracy'] * 100
            dl_enhanced = runner.results[dl_key]['enhanced']['overall_accuracy'] * 100
        else:
            dl_baseline = dl_enhanced = 0

        if ws_key in runner.results:
            ws_baseline = runner.results[ws_key]['baseline']['overall_accuracy'] * 100
            ws_enhanced = runner.results[ws_key]['enhanced']['overall_accuracy'] * 100
        else:
            ws_baseline = ws_enhanced = 0

        if th_key in runner.results:
            th_baseline = runner.results[th_key]['baseline']['overall_accuracy'] * 100
            th_enhanced = runner.results[th_key]['enhanced']['overall_accuracy'] * 100
        else:
            th_baseline = th_enhanced = 0

        print(f"{dataset.replace('_', ' ').title():<20} {dl_baseline:<12.1f} {dl_enhanced:<12.1f} "
              f"{ws_baseline:<12.1f} {ws_enhanced:<12.1f} "
              f"{th_baseline:<12.1f} {th_enhanced:<12.1f}")

    print("="*60)
    print("\nBased on thesis results (Chapter 8):")
    print("- All enhanced methods outperform baselines (p < 0.01)")
    print("- WS-GA achieves highest accuracy on all datasets")
    print("- DE-GA offers best speed-accuracy trade-off")
    print("- ILS-HED enables model enhancement without retraining")


if __name__ == '__main__':
    main()

