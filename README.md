# HSI Computational Intelligence

**Automated Design of Hyperspectral Image Segmentation Techmiques**

This repository contains the Python implementation accompanying the thesis. Three complementary
enhancement frameworks are evaluated against five standard benchmark datasets.

---

## Directory Structure

```
hsi-computation-intel/
├── main.py                        # CLI entry point (wraps main_experiment.py)
├── main_experiment.py             # Authoritative experiment runner (do not modify)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/                       # Place .mat dataset files here (not committed)
├── results/                       # Experiment output (not committed)
├── src/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # SegmentationMetrics, PairedTTest
│   │   └── visualization.py       # ResultVisualizer
│   ├── models/
│   │   ├── __init__.py
│   │   └── hed.py                 # HolisticallyNestedEdgeDetection (VGG16 backbone)
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── ils_hed.py             # Iterative Local Search + HED hyper-heuristic
│   │   ├── watershed_ga.py        # WS-GA: Genetic Algorithm for watershed
│   │   └── degree_ga.py           # DE-GA: Differential Evolution + GA thresholding
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py         # HyperspectralDataLoader
│       └── preprocessing.py       # Normalisation, PCA, band removal, augmentation
└── tests/
    ├── __init__.py
    ├── test_metrics.py
    ├── test_preprocessing.py
    ├── test_degree_ga.py
    ├── test_watershed_ga.py
    └── test_data_loader.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.8.  
> A GPU is optional but recommended for the ILS-HED experiments.

---

## Dataset Download

Place the downloaded `.mat` files in `data/raw/`.

| Dataset | File | Source |
|---|---|---|
| Indian Pines | `IndianPines.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pavia University | `PaviaU.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Salinas Valley | `Salinas.mat` | [EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Houston 2013 | `Houston.mat` | [IEEE GRSS Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=459) |
| Botswana | `Botswana.mat` | [NASA EO-1 Hyperion](https://earthobservatory.nasa.gov/) |

---

## Usage

### Run experiments via CLI (`main.py`)

```bash
# Show help
python main.py --help

# Run all datasets, all experiments
python main.py

# Run a single dataset
python main.py --dataset indian_pines

# Run a specific experiment type on a specific dataset
python main.py --dataset pavia_university --experiment watershed

# Custom data / results directories
python main.py --data-dir /path/to/data --results-dir /path/to/results
```

Available `--experiment` values: `deep_learning`, `watershed`, `thresholding`, `all` (default).  
Available `--dataset` values: `indian_pines`, `pavia_university`, `salinas_valley`, `houston`, `botswana`, `all` (default).

### Run Tests

```bash
pytest tests/
```

---

## Method Descriptions

### ILS-HED — Iterative Local Search with Holistically-Nested Edge Detection

A hyper-heuristic that iteratively selects the best combination of classical edge detectors
(Canny, Sobel, Laplacian, Gabor) to complement a frozen VGG16-based HED model.  
The search explores 53 configurations and terminates when no improvement is observed for 10 iterations.

### WS-GA — Watershed with Genetic Algorithm

A Genetic Algorithm jointly optimises five watershed pipeline parameters:
number of PCA components, Gaussian smoothing σ, gradient threshold τ, minimum region size s_min,
and downstream classifier type (SVM / RF / k-NN).  
Fitness = 0.55 × OA + 0.45 × AA − coverage_penalty − smoothness_penalty.

### DE-GA — Differential Evolution + GA Thresholding

A hybrid algorithm for multilevel histogram thresholding.  
A DE phase handles global exploration; a GA phase with single-point crossover and
Gaussian mutation handles local exploitation.  
Fitness is Otsu's between-class variance.

---

## Expected Results (Overall Accuracy)

| Dataset | Baseline HED | ILS-HED | Baseline WS | WS-GA | Baseline BT | DE-GA |
|---|---|---|---|---|---|---|
| Indian Pines | ~64 % | ~70.5 % | ~55 % | ~70.9 % | ~52 % | ~67.5 % |
| Pavia University | ~68 % | ~73.5 % | ~57 % | ~72.5 % | ~54 % | ~69.3 % |
| Salinas Valley | ~70 % | ~74.7 % | ~61 % | ~71.7 % | ~58 % | ~71.3 % |
| Houston | ~66 % | ~72.2 % | ~53 % | ~70.2 % | ~50 % | ~66.3 % |
| Botswana | ~67 % | ~72.8 % | ~56 % | ~72.6 % | ~53 % | ~68.8 % |

All enhancements are statistically significant at p < 0.01 (paired t-test, n = 10).

---

## Citation / Author

> *[Mzoxolo Mbini — replace with your full citation once the thesis is published]*  
> Thesis: "Optimisation of Computational Intelligence Techniques for Hyperspectral Image Segmentation"
