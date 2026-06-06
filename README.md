# DECODE: Empowering Chemical Structures with Biological Insights for Scalable Phenotypic Virtual Screening

## Overview

**DECODE** (**DE**composing **C**ellular **O**bservations of **D**rug **E**ffects) is a multi-modal disentanglement framework that empowers chemical representations with intrinsic biological semantics to enable structure-based *in silico* biological profiling.

Contemporary drug discovery faces a fundamental trade-off:

- **Structural screening** offers scalability (billions of molecules) but lacks functional biological context.
- **High-content phenotypic profiling** (transcriptomics, morphology) provides deep biological insights but is resource-intensive and noisy.

DECODE bridges this gap by leveraging paired transcriptomic (L1000) and morphological (Cell Painting) data as **privileged information** during training. Through **geometric signal disentanglement** and **contrastive alignment**, DECODE extracts a measurement-invariant biological fingerprint from chemical structures, enabling scalable identification of bioactive compounds with the accuracy of phenotypic assays — without requiring biological data at inference.

> **Paper:** *Empowering Chemical Structures with Biological Insights for Scalable Phenotypic Virtual Screening*
> Xiaoqing Lian, Pengsen Ma, Tengfeng Ma, Zhonghao Ren, Xibao Cai, Zhixiang Cheng, Bosheng Song, He Wang, Xiang Pan, Yangyang Chen, Sisi Yuan, Chen Lin
> Submitted to *Bioinformatics* (Oxford University Press)

---

## Key Features

- **Geometric Signal Disentanglement** — Orthogonality constraints decompose each modality into a shared response-related component and a modality-specific component, separating biological signals from experimental noise and batch effects.
- **Contrastive Cross-Modal Alignment** — InfoNCE-based alignment transfers biological response information from transcriptomic and morphological views into the chemical representation.
- **Modality-Mask Self-Reconstruction** — Training with randomly masked modalities improves robustness to missing biological profiles at inference.
- **Learning Using Privileged Information (LUPI)** — Biological profiles are used only during training; inference requires only chemical structures (SMILES).
- **Flexible Backbone** — Supports MolFormer and VideoMol as molecular encoders.
- **Three Adaptive Inference Protocols** — Zero-shot functional retrieval, modality-flexible MoA prediction, and profile-guided virtual screening.

---

## Framework Architecture

DECODE operates on a **Learning Using Privileged Information (LUPI)** paradigm:

1. **Training:** The model sees Chemical Structure ($x_d$), Gene Expression ($x_g$), and Cell Morphology ($x_m$). It decomposes each modality into shared and modality-specific components via **Orthogonal Disentanglement**, aligns shared components via **Contrastive Learning**, and reconstructs missing modalities via **Modality-Mask Training**.
2. **Inference:** The model requires **only Chemical Structure** ($x_d$). It generates a biologically informed representation for downstream tasks.

### Three Inference Protocols

| Protocol | Task | Input | Description |
|---|---|---|---|
| **I** | Zero-Shot Functional Retrieval | Chemical only | Retrieve functionally similar drugs (same MOA) without supervision |
| **II** | Modality-Flexible MoA Prediction | Chemical + available profiles | Predict MOA under various missing-modality scenarios |
| **III** | Profile-Guided Virtual Screening | Chemical only | Generate-Refine-Enhance pipeline for identifying novel bioactive compounds |

---

## Project Structure

```
DECODE/
├── models/distangle_multimodal/   # Core DECODE model & training scripts
├── virtual_screening/              # Downstream task modules (retrieval, MOA, screening)
├── DModule/                        # PyTorch Lightning DataModules
├── Molformer/                      # Local MolFormer encoder implementation
├── utils/                          # Utility functions (metrics, data splitting, etc.)
├── preprocessed_data/              # Preprocessed datasets (LINCS, CDRP, Virtual Screening)
└── requirements.txt
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/lian-xiao/DECODE.git
cd DECODE
pip install -r requirements.txt
```

### 2. Stage 1: Pretraining (Geometric Disentanglement & Alignment)

Train the DECODE backbone to learn measurement-invariant biological fingerprints using paired data (Structure + Transcriptomics + Morphology).

```bash
# Using MolFormer backbone
python models/distangle_multimodal/train_multimodal_stage1_predictor.py \
    --config models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml

# Using VideoMol backbone
python models/distangle_multimodal/train_multimodal_stage1_predictor.py \
    --config models/distangle_multimodal/config_distatngle_multimodal_moa_predictor_videomol.yaml
```

### 3. Stage 2: Fine-tuning (Task-Specific Optimization)

Fine-tune the pretrained model for downstream tasks using the two-stage training script.

```bash
python models/distangle_multimodal/train_multimodal_two_stage_predictor.py \
    --stage1_config models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml \
    --stage2_config models/distangle_multimodal/config_distatngle_multimodal_moa_predictor2.yaml \
    --split_indices 0 1 2 3 4
```

### 4. Downstream Applications

**Protocol I: Zero-Shot Functional Retrieval**

Evaluate the structure encoder's ability to retrieve functionally similar drugs without supervision.

```bash
python virtual_screening/no_training_retrieval_test.py
```

**Protocol II: MOA Classification (Modality-Flexible)**

Fine-tune for Mechanism of Action prediction under various modality-availability settings.

```bash
python virtual_screening/train_moa_classification.py \
    --config virtual_screening/config_virtual_screening.yaml
```

**Protocol III: Virtual Screening (Binary Classification)**

Run the Generate-Refine-Enhance pipeline for identifying novel bioactive compounds.

```bash
python virtual_screening/train_virtual_screening.py \
    --config virtual_screening/config_virtual_screening.yaml
```

**Pathway Prediction (Multi-Label)**

Predict compound-pathway associations on MCELC dataset.

```bash
python virtual_screening/train_pathway_prediction.py \
    --config virtual_screening/config_virtual_screening.yaml
```

---

## Datasets

| Dataset | Modalities | Tasks |
|---|---|---|
| **LINCS Pilot1** | Chemical + L1000 + Cell Painting | Pretraining, MOA classification, Retrieval |
| **CDRP-BBBC047-Bray** | Chemical + L1000 + Cell Painting | Pretraining, MOA classification, Retrieval |
| **PRISM Cancer** | Chemical (+ generated profiles) | Binary cancer activity screening |
| **MCELC** | Chemical (+ generated profiles) | Multi-label pathway prediction |
| **BACE1 / COX-1 / COX-2 / EP4** | Chemical (+ generated profiles) | Binary target-specific screening |

---

## Performance Highlights

- **Functional Retrieval:** DECODE captures functional similarity beyond structural embeddings, enabling zero-shot retrieval of drugs with the same MOA.
- **Modality-Flexible Prediction:** Stable performance across complete-data, morphology-missing, gene-expression-missing, and dual-profile-missing settings.
- **Virtual Screening:** Improved early-recognition metrics (EF1%, BEDROC20) over molecular and multimodal baselines under chemically informed splits.
- **Tissue-Stratified Screening:** DECODE increases AP, BEDROC20, and EF1% across diverse cancer tissue types in PRISM evaluation.

---

## Citation

If you use DECODE in your research, please cite our paper:

```bibtex
@article{Lian2026DECODE,
  title={Empowering Chemical Structures with Biological Insights for Scalable Phenotypic Virtual Screening},
  author={Lian, Xiaoqing and Ma, Pengsen and Ma, Tengfeng and Ren, Zhonghao and Cai, Xibao and Cheng, Zhixiang and Song, Bosheng and Wang, He and Pan, Xiang and Chen, Yangyang and Yuan, Sisi and Lin, Chen},
  journal={Bioinformatics},
  year={2026},
  publisher={Oxford University Press}
}
```

---

## Contact

For questions regarding the code or datasets, please open an issue or contact:
- Xiaoqing Lian: [lianxiaoqing@hnu.edu.cn](mailto:lianxiaoqing@hnu.edu.cn)
- Yangyang Chen: [chen.yangyang.xp@alumni.tsukuba.ac.jp](mailto:chen.yangyang.xp@alumni.tsukuba.ac.jp)
- Sisi Yuan: [sisiyuan@hkbu.edu.hk](mailto:sisiyuan@hkbu.edu.hk)
- Chen Lin: [cheyenne.lin@foxmail.com](mailto:cheyenne.lin@foxmail.com)

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62432011, 62450002).
