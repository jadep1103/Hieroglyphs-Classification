# Egyptian Hieroglyphs Classification with CNNs and Transformers

## Project Overview

This academic project develops and compares multiple Deep Learning models for classifying Egyptian hieroglyphs. We explore different architectures (CNN, ResNet, Inception, Vision Transformer) and training strategies including training from scratch, using pre-trained models with fine-tuning, and feature extraction approaches.

The project aims to facilitate the identification of hieroglyphic symbols to support more complex analyses such as reconstructing erased texts, studying semantic structures, and assisting in translation tasks.

## 📊 Results Summary

| Model | Accuracy | F1-macro | Precision | Recall |
|-------|----------|-----------|-----------|---------|
| ResNet18 (frozen) | 98.7% | 94.8% | 95.3% | 94.8% |
| Inception-v3 (frozen) | 97.8% | 91.6% | 93.1% | 91.4% |
| ResNet50 (frozen) | 96.2% | 89.6% | 91.3% | 90.2% |
| ViT Fine-tuned | 94.7% | 94.2% | 94.6% | 94.0% |
| ViT Frozen | 91.1% | 90.4% | 91.0% | 89.9% |
| ViT from Scratch | 87.9% | 74.2% | 75.1% | 78.4% |
| CustomCNN | 62.7% | 32.1% | 35.2% | 35.3% |

## 🛠️ Tech Stack
PyTorch • torchvision • scikit-learn
OpenCV • Pillow (PIL) • NumPy • Pandas • Matplotlib • Seaborn
## 📁 Repository Structure

```
├── dataset_download_pretrained_classifier/    # Initial experiments with multiple models
│   ├── data_augmentation.py                  # Data augmentation implementation
│   ├── dataset.py                           # Dataset loading and preprocessing
│   ├── download_dataset.py                  # Dataset download utilities
│   ├── main.py                              # Main training script
│   ├── models.py                            # Model architectures
│   └── train.py                             # Training functions
│
├── resnet_and_inception/                     # ResNet and Inception experiments
│   ├── dataaugmentation.py                  # Data augmentation for ResNet/Inception
│   ├── dataset.py                          # Dataset utilities
│   ├── filter.py                           # Dataset filtering (top 40 classes)
│   ├── main.py                             # Main script for ResNet/Inception
│   ├── models.py                           # ResNet50 and Inception-v3 models
│   ├── train.py                            # Training with advanced metrics
│   ├── history_*.csv                       # Training histories
│   └── report_*.csv                        # Classification reports
│
├── vision-transformer/                       # Vision Transformer experiments
│   ├── dataaugmentation.py                 # Data augmentation for ViT
│   ├── dataset.py                          # Dataset loading
│   ├── filter.py                           # Dataset filtering
│   ├── main.py                             # Main ViT training script
│   ├── models.py                           # ViT architectures (custom + pretrained)
│   ├── train.py                            # ViT training functions
│   ├── history_*.csv                       # ViT training histories
│   └── report_*.csv                        # ViT classification reports
│
└── Projet.pdf                              # Complete project report
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision
pip install pillow tqdm pandas matplotlib seaborn scikit-learn
pip install roboflow  # For dataset download (optional)
```

### Dataset Setup

The project uses the "EgyptianHieroglyphDataset Computer Vision Project" with 170 classes annotated according to Gardiner's classification. We selected the top 40 most represented classes for our experiments.

1. **Automatic Download** (if roboflow is available):
```bash
python dataset_download_pretrained_classifier/download_dataset.py
```

2. **Manual Download**:
   - Visit: https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset
   - Extract to `EgyptianHieroglyphDataset-1/`

3. **Dataset Filtering** (to get top 40 classes):
```bash
python resnet_and_inception/filter.py
```

4. **Data Augmentation** (target: 200 images per class):
```bash
python resnet_and_inception/dataaugmentation.py
```

### Running Experiments

#### 1. ResNet and Inception Models
```bash
cd resnet_and_inception
python main.py
```

#### 2. Vision Transformers
```bash
cd vision-transformer
python main.py
```

#### 3. Multiple Model Comparison
```bash
cd dataset_download_pretrained_classifier
python main.py
```

## 🏗️ Architecture Details

### 1. CustomCNN (from scratch)
- 3 convolutional blocks (32, 64, 128 channels)
- BatchNorm2d + ReLU + AvgPool2d
- AdaptiveAvgPool2d + Dropout(0.3)
- Classification head: 128 → 64 → 40

### 2. ResNet18/50 (pre-trained, frozen backbone)
- Pre-trained on ImageNet
- Only final classification layer fine-tuned
- ResNet18: 512 → 40 features
- ResNet50: 2048 → 40 features

### 3. Inception-v3 (pre-trained, frozen backbone)
- Pre-trained on ImageNet with auxiliary head
- Factorized convolutions for efficiency
- Multi-scale feature extraction
- 2048 → 40 classification head

### 4. Vision Transformer Variants

**Custom ViT (from scratch):**
- 16×16 patches → 256 dimensions
- 5 Transformer layers, 8 attention heads
- CLS token + learned positional embeddings
- LayerNorm + Linear(256 → 40)

**ViT-B/16 (pre-trained):**
- Pre-trained on ImageNet-21k
- Two variants: frozen backbone vs. full fine-tuning
- 768 → 40 classification head

## 📈 Training Configuration

### Data Augmentation Strategy
- **Target**: 200 images per class
- **Transformations**:
  - Random rotation: ±16°
  - Translation: 0-20% on x,y axes
  - Scale: 65-100% of original size
  - Brightness/contrast adjustment
  - Noise background generation

### Hyperparameters

| Model | Learning Rate | Epochs | Batch Size | Optimizer | Scheduler |
|-------|---------------|---------|------------|-----------|-----------|
| CustomCNN | 5e-4 | 12 | 32 | Adam | RLRP |
| ResNet18 | 1e-3 | 20 | 64 | Adam | - |
| ResNet50 | 1e-3 | 15 | 64 | Adam | - |
| Inception | 1e-3 | 15 | 64 | Adam | - |
| ViT scratch | 1e-4 | 20 | 32 | Adam | - |
| ViT frozen | 1e-3 | 15 | 64 | AdamW | - |
| ViT finetune | 1e-5 | 30 | 32 | AdamW | RLRP |

## 🔍 Key Findings

### Best Performing Models
1. **ResNet18 (frozen)**: 98.7% accuracy - Best overall performance
2. **Inception-v3**: 97.8% accuracy - Excellent multi-scale feature extraction  
3. **ViT Fine-tuned**: 94.7% accuracy - Best transformer approach

### Important Observations
- **Pre-trained models significantly outperform** models trained from scratch
- **Data augmentation is crucial** for good generalization (5+ point improvement)
- **Feature extraction approach** (frozen backbone) works quite well
- **Custom CNN struggles** with limited data despite regularization

### Ablation Study Results
- Removing data augmentation: -2.5% accuracy, significant F1 drop
- Reducing ViT layers: Minimal accuracy change but worse precision/recall
- Test-time augmentation: Implementation issues led to poor results

## 📊 Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-score (macro)**: Balanced performance across all classes
- **Precision (macro)**: Average precision across classes
- **Recall (macro)**: Average recall across classes
- **Confusion matrices**: Per-class performance analysis

## 🛠️ Implementation Details

### Multi-GPU Training
The codebase supports multi-process training for faster experimentation:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

### Logging and Visualization
- Training curves automatically saved to `logs/`
- Confusion matrices generated for each model
- Classification reports in CSV format
- Integration with Weights & Biases (optional)

## 📚 References

1. [Egyptian Hieroglyph Dataset](https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset) - 3,584 images, 170 classes
2. [Gardiner's Sign List](https://www.egyptianhieroglyphs.net/gardiners-sign-list/) - Classification system
3. Dosovitskiy et al. (2020) - Vision Transformer architecture

## 🤝 Contributing

This is an academic project completed as part of coursework. The repository serves as a reference for:
- Comparative analysis of CNN vs. Transformer architectures
- Transfer learning strategies for specialized domains
- Data augmentation techniques for limited datasets
- Multi-class classification evaluation

## 👥 Authors

- Jade Piller-Cammal 
- Estelle Tournassat 
- Théo Parris
- Alban Sarrazin 

## 📄 License

This project is for educational purposes. The dataset is provided by Roboflow under their respective license terms.

## 🚧 Project Status: On Standby - Major Refactor Planned

**⚠️ Important Notice**: This repository represents the initial academic submission. While the research methodology and results are interesting, the codebase has significant technical debt that needs addressing.

### Current Issues Identified
- **Code duplication** across multiple folders
- **Poor code organization** with inconsistent naming and structure
- **Hardcoded configurations** scattered throughout
- **Error handling** and missing proper logging
- **Mid reproducibility controls** (seeds, environment management)
- **Manual experiment tracking** instead of automated solutions

### 🛠️ Planned Improvements

#### **1. Code Refactor**

#### **2. Dataset Expansion**
- Investigate larger hieroglyph datasets beyond current 40 classes
- Explore museum collections and archaeological databases
- Research synthetic data generation approaches

#### **3. Technical Upgrades**
- Experiment with modern architectures 
- Experiment tracking with MLflow/WandB
- Production deployment with API and web interface ? 

---

**Status**: Academic submission - major refactor incoming  
