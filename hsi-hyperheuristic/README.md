---

# Hyper-Heuristic Framework for Hyperspectral Image Segmentation

## 1. Overview

Research-grade implementation of the hyper-heuristic framework described in:

**Optimisation of Computation Intelligence Techniques for Image Segmentation**
PhD Thesis, University of Pretoria, 2024

**Author:** Mzoxolo Mbini
**Version:** 1.0 (Research Validation)
**DOI:** [PLACEHOLDER - update before submission]

---

## 1.1 What This Code Does

* Automatically discovers optimal hyperspectral segmentation pipelines using grammar-guided genetic programming
* Adapts low-level heuristics (SS-PSO, gradient operators, clustering) using spectralâspatial meta-features
* Achieves state-of-the-art segmentation with mIoU = 0.874 using 260Ã fewer parameters than deep learning baselines
* Validates all six research hypotheses (H1âH6) via bootstrap-based statistical testing

---

## 1.2 Key Achievements

| Metric                      | Thesis Value | Implementation |
| --------------------------- | ------------ | -------------- |
| Segmentation Accuracy       | mIoU = 0.874 | â              |
| Parameter Efficiency        | 0.047M       | â              |
| Training Time               | 14.3 minutes | â              |
| Energy Efficiency           | 7.1Ã         | â              |
| Cross-Domain Generalization | 90.3%        | â              |
| Statistical Significance    | p < 0.001    | â              |

---

## 2. Installation and Dependencies

### 2.1 System Requirements

**Hardware**

* GPU: 4Ã NVIDIA A100 (40GB) recommended
* CPU: 64+ cores
* RAM: 512GB
* Storage: 2TB NVMe + 10TB NAS

**Software**

* Ubuntu 20.04 LTS
* CUDA 11.3
* Python 3.8.12

---

### 2.2 Docker Environment

```dockerfile
FROM nvidia/cuda:11.3-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3-venv \
    libopencv-dev libgdal-dev git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install \
    scikit-learn==1.1.2 \
    scikit-image==0.19.3 \
    scipy==1.9.1 \
    numpy==1.23.1 \
    matplotlib==3.5.2 \
    seaborn==0.11.2 \
    ray==2.0.0 \
    opencv-python==4.6.0 \
    pandas==1.4.3 \
    tqdm==4.64.1 \
    h5py==3.7.0 \
    scikit-fuzzy==0.4.2 \
    networkx==2.8.8 \
    pygad==3.2.0 \
    deap==1.3.3

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p /workspace/data /workspace/results /workspace/cache

ENV PYTHONPATH=/workspace
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

CMD ["python", "hh_hed.py", "--help"]
```

**Build**

```bash
docker build -t hsi-hyperheuristic:v1.0 .
```

**Run**

```bash
docker run --gpus all -v /data:/data -v /results:/results hsi-hyperheuristic:v1.0
```

---

### 2.3 Python Dependencies

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Data Preparation

### 3.1 Required Datasets

| Dataset          | Source           | License       | Directory             |
| ---------------- | ---------------- | ------------- | --------------------- |
| Indian Pines     | Purdue MultiSpec | Public Domain | data/Indian_Pines     |
| Pavia University | IEEE DASE        | CC BY-NC-SA   | data/Pavia_University |
| Salinas          | USGS             | Public Domain | data/Salinas          |
| Houston          | IEEE GRSS        | Research      | data/Houston          |
| Botswana         | USGS             | Public Domain | data/Botswana         |

---

### 3.2 Directory Structure

```text
/workspace/
âââ hh_hed.py
âââ requirements.txt
âââ Dockerfile
âââ config.json
âââ LICENSE
âââ README.md
âââ setup.py
âââ data/
âââ src/
âââ results/
âââ cache/
âââ tests/
```

---

### 3.3 Preprocessing Pipeline

Seven-stage automated pipeline:

1. Radiometric correction
2. Bad band removal
3. Atmospheric correction
4. Spectralâspatial denoising
5. Dimensionality reduction
6. Patch extraction
7. Data augmentation

---

## 4. Usage

### 4.1 Reproduce Thesis Results

```bash
docker run --gpus all -it -v $(pwd)/data:/data -v $(pwd)/results:/results hsi-hyperheuristic:v1.0
python hh_hed.py --mode full_validation --dataset Indian_Pines --n_runs 10
python hh_hed.py --mode generate_report --results_path /results/evaluation_results.json
```

---

### 4.2 Command-Line Interface

```bash
python hh_hed.py --help
```

Modes include full_validation, train_only, evaluate_only, baseline, generate_report, verify_data.

---

### 4.3 Python API Example

```python
from src.framework.hyperheuristic import HyperHeuristicFramework
from src.utils.config_parser import Config
from src.utils.data_loader import HyperspectralDataset

config = Config.from_json("config.json")
framework = HyperHeuristicFramework(config)

train_data = HyperspectralDataset(config, "Indian_Pines", "train")
val_data = HyperspectralDataset(config, "Indian_Pines", "val")

framework.train(train_data, val_data)
```

---

## 5. Configuration

All hyperparameters match the thesis exactly. See `config.json` for full specification.

---

## 6. Output Files

Generated in `./results/`:

* evolution_results.json
* evaluation_results.json
* pareto_frontier.png
* statistical_report.pdf
* segmentation maps

---

## 7. Reproduction Checklist

* 4Ã NVIDIA A100 GPUs
* Ubuntu 20.04, CUDA 11.3
* Python 3.8.12
* Random seed = 42
* Set PYTHONHASHSEED=42 in shell BEFORE launching Python (e.g., `export PYTHONHASHSEED=42`)
* All datasets verified by checksum

---

## 8. Contact

**Author:** Mzoxolo Mbini
Email: [u16350244@tuks.co.za](mailto:u16350244@tuks.co.za)
Institution: University of Pretoria

---

## 9. Version History

* v1.0 (2024-02-15): Thesis-aligned release
* v1.01: Multi-modal fusion
* v1.1: Multi-temporal extension
* v2.0: Foundation model integration

---

## 10. Architecture Overview

```text
Hyperspectral Input
 â Preprocessing
 â Meta-Feature Extraction
 â Policy Network
 â Low-Level Heuristics
 â Segmentation Fusion
 â Output Map
```

---

Last updated: 2024-02-15
Version: 1.0
DOI: [PLACEHOLDER - update before submission]
