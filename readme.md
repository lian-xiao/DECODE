# DECODE: Disentangled Multimodal Model for Virtual Screening

## Overview

Integrating diverse cellular profiles (e.g. morphology and gene expression) offers a holistic view for predicting a compound's mechanism of action (MOA). However, primary biological signals are entangled with measurement-specific artifacts, and their indiscriminate fusion yields a confounded representation that impairs model generalization. Inspired by the principle that a drug's primary effects should elicit a consistent signal across assays, we developed DECODE, an autoencoder that learns a measurement-agnostic consensus representation of drug action while explicitly preserving modality-specific features. By anchoring biological profiles with known compound identities, DECODE aligns them into a unified latent space, robustly disentangling the consensus signal from technical variations. The resulting embeddings enable DECODE to outperform baseline methods in MOA prediction. In zero-shot MOA retrieval, DECODE reliably identifies functionally similar compounds that are structurally dissimilar even when only their chemical structure is available. Furthermore, by integrating simulated biological profiles for unprofiled compounds, DECODE enhances virtual screening, increasing the hit rate for novel anticancer drugs by 6-fold in external validationand improving the prediction accuracy of their action pathways in lung cancer.

## Key Features

- **Disentangled Multimodal Learning**: Separates complementary and unique modality-specific representations
- **Virtual Screening Module**: Validation framework for screening efficacy assessment
- **End-to-End Pipeline**: Integrated training and evaluation workflows
- **Extensible Architecture**: Modular design for easy adaptation to new modalities and tasks

## Project Structure

```
DECODE/
├── models/distangle_multimodal/
│   ├── train.py                 # Training script for multimodal disentanglement
│   ├── model.py                 # Core model architecture
│   └── config.yaml              # Model configuration
├── virtual_screening/
│   ├── train.py                 # Virtual screening model training
│   ├── model.py                 # Core model architecture
│   └── config.yaml              # Screening configuration
├── DModule/                     # DataModule
├── preprocessed_data/           # Dataset directory (to be populated)
├── results/                     # Output directory for results
└── requirements.txt             # Dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Two Stage Training DECODE

```bash
cd models/distangle_multimodal
python train_multimodal_two_stage_predictor.py
```

### Virtual Screening Validation

```bash
cd virtual_screening
python train_virtual_screening.py
python train_moa_classification.py
python train_pathway_prediction.py
```

## Main Components

### Disentangled Multimodal Module (`models/distangle_multimodal/`)

- Learns disentangled representations from multiple input modalities
- Separates shared and modality-specific information
- Improves generalization and interpretability

### Virtual Screening Module (`virtual_screening/`)

- Validates model predictions on real drug discovery tasks
- Evaluates screening performance metrics
- Supports benchmark comparison

## Requirements

- Python 3.8+
- PyTorch >= 2.2
- NumPy, Pandas, scikit-learn
- See `requirements.txt` for complete dependencies

## Citation

If you use DECODE in your research, please cite:

```bibtex
@article{decode2024,
  title={A Measurement-Invariant Functional Fingerprint of Drug Action through Disentanglement of Multi-Modal Cellular Profiles},
  author={Xiaoqing Lian , Xiangxiang Zeng , Pengsen Ma , Tengfei Ma , Xibao Cai , Zhixiang Cheng , He Wang , Xiang Pan , Quan Zou , Chen Lin},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on the project repository.
