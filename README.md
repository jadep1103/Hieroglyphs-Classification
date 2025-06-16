# 🐍 Egyptian Hieroglyphs Classification  
_Deep Learning-based symbol recognition with CNNs and Vision Transformers_ 
> Project for the GLO-7030 Deep Learning course – Université Laval

---

## 📌 Overview
This project explores the classification of **Egyptian hieroglyphs** using modern **deep learning models**. We implement and evaluate both **CNNs** and **Vision Transformers (ViT)** for recognizing symbols annotated with the Gardiner sign list.

Given the challenges posed by a limited and imbalanced dataset, we investigate several training strategies, including:
- Data augmentation
- Transfer learning with pretrained models
- Fine-tuning vs. feature extraction
- An ablation study to assess each component’s impact

---

## 🔍 Objective
Develop a robust image classifier capable of identifying Egyptian hieroglyphs, laying the groundwork for potential downstream tasks such as **translation assistance**, **theme extraction**, and **semantic reconstruction** of ancient texts.

---

## 📊 Dataset
We use the [Egyptian Hieroglyph Dataset](https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset), containing 3,584 labeled images across 170 classes. For better class balance, only the top **40 most frequent classes** were selected. All images were resized to **224×224** pixels.

To improve generalization, we applied data augmentation with:
- Random rotation (±16°)
- Translation (up to 20%)
- Scaling down to 80%
- Brightness and contrast variation

---

## 🧠 Models Implemented

| Model                  | Pretrained | Fine-Tuned     | Accuracy | F1 Score |
|-----------------------|------------|----------------|----------|----------|
| Custom CNN            | ❌         | –              | 62.7%    | 32.1%    |
| ResNet18              | ✅         | Head only      | 98.7%    | 94.8%    |
| ResNet50              | ✅         | Head only      | 96.2%    | 89.6%    |
| Inception-v3          | ✅         | Head only      | 97.8%    | 91.6%    |
| ViT (from scratch)    | ❌         | –              | 87.9%    | 74.2%    |
| ViT (frozen)          | ✅         | Head only      | 91.1%    | 90.4%    |
| ViT (fine-tuned)      | ✅         | Full model     | 94.7%    | 94.2%    |

> ✅ Best overall: **ViT pretrained with full fine-tuning**

---

## 🔬 Ablation Study
We evaluated the effect of:
- Disabling data augmentation
- Reducing the number of Transformer layers
- Adding test-time augmentation (TTA)

🧪 **Findings**:
- No augmentation led to lower F1 and recall
- Smaller models generalized poorly
- TTA failed (1.5% accuracy), likely due to implementation bugs

---

## ⚙️ Training Setup

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Epochs: 12–30 (depending on model)
- Batch size: 32 or 64
- Some models used `ReduceLROnPlateau` scheduler

---

## 👥 Authors
- Jade Piller-Cammal  
- Estelle Tournassat  
- Théo Parris  
- Alban Sarrazin  

---

## 📜 References
- 📚 Dataset: [Roboflow – Egyptian Hieroglyphs](https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset)  
- 🔠 Labels: [Gardiner's Sign List](https://www.egyptianhieroglyphs.net/gardiners-sign-list/)

---
