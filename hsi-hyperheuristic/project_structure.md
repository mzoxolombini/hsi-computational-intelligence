hsi-hyperheuristic/
âââ README.md
âââ requirements.txt
âââ setup.py
âââ Dockerfile
âââ config.json
âââ run.py                    # Main CLI entry point
âââ data/                     # Datasets (auto-downloaded)
âââ checkpoints/              # Model checkpoints (e.g., gradient_v6.2.pt)
âââ src/
â   âââ __init__.py
â   âââ config/               # Configuration management
â   â   âââ __init__.py
â   â   âââ config_loader.py
â   â   âââ constants.py
â   â   âââ hyperparameters.py
â   âââ data/                 # Data handling
â   â   âââ __init__.py
â   â   âââ dataset_loader.py
â   â   âââ preprocessing.py
â   â   âââ augmentations.py
â   â   âââ meta_features.py
â   âââ llhs/                 # Low-Level Heuristics
â   â   âââ __init__.py
â   â   âââ base.py
â   â   âââ sspso.py
â   â   âââ gradient.py          # HolisticGradientOperator
â   â   âââ clustering.py
â   â   âââ watershed.py
â   â   âââ mrf.py
â   â   âââ cnn_refine.py
â   â   âââ crf.py
â   âââ gp/                   # Genetic Programming
â   â   âââ __init__.py
â   â   âââ grammar.py
â   â   âââ individual.py
â   â   âââ evolution.py
â   â   âââ evaluation.py
â   â   âââ pareto_front.py
â   âââ policy/               # Policy Network
â   â   âââ __init__.py
â   â   âââ network.py
â   â   âââ trainer.py
â   â   âââ selector.py
â   âââ transfer/             # Transfer Learning Components (NEW)
â   â   âââ __init__.py
â   â   âââ adapter.py        # GradientTransferAdapter
â   â   âââ task_losses.py    # Task-specific loss functions
â   âââ deployment/           # Edge Deployment Orchestration (NEW)
â   â   âââ __init__.py
â   â   âââ orchestrator.py   # EdgeOrchestrator for resource management
â   â   âââ modality_router.py
â   âââ framework/            # Main Hyper-Heuristic Framework
â   â   âââ __init__.py
â   â   âââ hyperheuristic.py
â   â   âââ trainer.py
â   â   âââ evaluator.py
â   â   âââ segmenter.py
â   âââ utils/                # Utilities
â   â   âââ __init__.py
â   â   âââ reproducibility.py
â   â   âââ stats.py
â   â   âââ visualization.py
â   â   âââ metrics.py
â   â   âââ logger.py
â   â   âââ profiler.py
â   âââ experiments/          # Experiment Scripts
â       âââ __init__.py
â       âââ run_experiment.py
â       âââ baseline_comparison.py
âââ scripts/                  # Shell and CLI Scripts
â   âââ download_datasets.sh
â   âââ run_docker.sh
â   âââ run_experiments.sh
â   âââ adapt_to_domain.py    # CLI for Transfer Learning (NEW)
âââ tests/                    # Unit Tests
â   âââ __init__.py
â   âââ test_data.py
â   âââ test_llhs.py
â   âââ test_transfer.py      # Transfer Adapter Tests (NEW)
â   âââ test_framework.py
âââ notebooks/                # Jupyter Notebooks
â   âââ 01_data_exploration.ipynb
â   âââ 02_results_analysis.ipynb
â   âââ 03_transfer_learning.ipynb  # Demo Notebook (NEW)
âââ results/                  # Output Directory (Auto-Created)
