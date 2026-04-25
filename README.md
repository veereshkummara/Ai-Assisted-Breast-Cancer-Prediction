# 🩺 AI-Assisted Breast Cancer Diagnosis
### Vision Transformer + U-Net · Explainable AI · Real-Time Streamlit App

---

## 📂 PROJECT FOLDER STRUCTURE

```
breast_cancer_ai/
│
├── app.py                      ← Streamlit application (main entry point)
├── config.py                   ← All hyperparameters & paths
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   └── vit_unet.py             ← ViT encoder + U-Net decoder
│
├── explainability/
│   ├── __init__.py
│   ├── gradcam.py              ← Grad-CAM & Grad-CAM++
│   └── attention_maps.py       ← Attention rollout, per-head maps
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py              ← Dice, IoU, Precision, Recall, AUC
│   ├── preprocessing.py        ← CLAHE, augmentations, tensor utils
│   └── visualization.py        ← Heatmap overlays, comparison grids
│
├── training/
│   ├── __init__.py
│   ├── dataset.py              ← BUS dataset loader (PyTorch Dataset)
│   └── train.py                ← Full training loop
│
├── weights/                    ← Saved .pth model weights go here
│   └── best_model.pth          (created after training)
│
├── data/                       ← Dataset goes here (see step 3)
│   └── BUS_dataset/
│       ├── benign/
│       ├── malignant/
│       └── normal/
│
└── sample_images/              ← Test images for quick demo
```

---

## 🗄️ DATASET — BUSI (Breast Ultrasound Images)

**Name:** Breast Ultrasound Images Dataset (BUSI)  
**Source:** Kaggle → https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset  
**License:** CC BY 4.0 (free for research)  
**Size:** ~800 images across 3 classes

### Class breakdown
| Class     | Images | Description                          |
|-----------|--------|--------------------------------------|
| Normal    |   133  | No lesion, no mask                   |
| Benign    |   437  | Non-cancerous mass with mask         |
| Malignant |   210  | Cancerous tumour with mask           |

### Download & place
```bash
# After downloading from Kaggle, extract to:
breast_cancer_ai/data/BUS_dataset/

# Expected structure:
data/BUS_dataset/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── benign (2).png
│   ├── benign (2)_mask.png
│   └── ...
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   └── ...
└── normal/
    ├── normal (1).png
    └── ...    (no masks for normal class)
```

---

## ⚙️ STEP-BY-STEP SETUP (Local Server)

### Step 1 — Python environment
```bash
# Python 3.9 or 3.10 recommended
python --version

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 2 — Install dependencies
```bash
cd breast_cancer_ai
pip install -r requirements.txt

# If you have a CUDA GPU (recommended for training):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Download & place dataset
```bash
# Install Kaggle CLI
pip install kaggle

# Download BUSI dataset (requires Kaggle API key)
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d data/BUS_dataset/
```

### Step 4 — Quick sanity check
```bash
python models/vit_unet.py
# Expected output:
# Seg output : torch.Size([2, 1, 224, 224])
# Cls output : torch.Size([2, 3])
# Attn shape : torch.Size([2, 12, 197, 197])
```

### Step 5 — Train the model
```bash
python training/train.py
# Training will:
# - Auto-load timm ViT-Base/16 pretrained weights (ImageNet)
# - Fine-tune on BUSI dataset
# - Save best model to weights/best_model.pth
# - Log training history to logs/history.json
#
# Typical training time:
# - GPU (RTX 3080): ~3-5 min/epoch → 50 epochs ~3 hrs
# - CPU only: ~30-60 min/epoch (use fewer epochs)
```

### Step 6 — Run the Streamlit app
```bash
streamlit run app.py

# App will open at: http://localhost:8501
```

---

## 🏗️ ARCHITECTURE — DETAILED EXPLANATION

### 1. Patch Embedding
```
Input: [B, 3, 224, 224]
→ Split into 14×14 = 196 patches of 16×16 pixels
→ Each patch → linear projection → 768-dim vector
→ Add [CLS] token + positional embedding
Output: [B, 197, 768]
```

### 2. ViT Encoder (12 Transformer Blocks)
Each block:
```
x → LayerNorm → MultiHeadAttention → + residual
  → LayerNorm → MLP (GELU) → + residual
```
Multi-scale feature tapping at blocks **2, 5, 8, 11**

### 3. U-Net Decoder
```
ViT tokens [B,196,768] → reshape → [B,768,14,14]   (bridge)
                                        ↓
                               DecoderBlock × 4
                               skip: proj(block_8) → 512ch
                               skip: proj(block_5) → 256ch
                               skip: proj(block_2) → 128ch
                                        ↓
                               [B, 32, 224, 224]
                                        ↓
                               Conv1×1 → [B, 1, 224, 224]  ← segmentation
```

### 4. Classification Head
```
CLS token [B,768] → LayerNorm → Linear(256) → GELU → Dropout → Linear(3)
```

---

## 🧠 EXPLAINABILITY METHODS

### Grad-CAM
- Hooks into the **bridge Conv2D** (bottleneck between encoder & decoder)
- Backpropagates class score → computes gradient w.r.t. feature maps
- Weights channels by global average pooling of gradients
- Output: spatial importance map at 14×14, upsampled to 224×224

### Grad-CAM++
- Extended Grad-CAM using second-order gradients (alpha weights)
- Better localization for multiple instances

### ViT Attention Rollout (Abnar & Zuidema, 2020)
- Recursively multiplies attention matrices across all 12 blocks
- Adds residual (identity) at each layer: A̅ = 0.5·A + 0.5·I
- Discards lowest 90% of attention for cleaner maps
- Output: 14×14 spatial map from CLS→patch attention

### Raw Last-Block Attention
- Direct CLS→patch attention from block 12
- Shows what the model "looks at" for classification

### Combined Map
- Weighted average of Grad-CAM + Attention Rollout
- Best for clinical interpretation

---

## 📊 EVALUATION METRICS

| Metric      | Formula                        | Target |
|-------------|--------------------------------|--------|
| Dice        | 2·\|A∩B\| / (\|A\|+\|B\|)     | > 0.80 |
| IoU/Jaccard | \|A∩B\| / \|A∪B\|             | > 0.70 |
| Precision   | TP / (TP+FP)                   | > 0.80 |
| Recall      | TP / (TP+FN)                   | > 0.80 |
| Specificity | TN / (TN+FP)                   | > 0.85 |
| AUC-ROC     | Area under ROC curve           | > 0.90 |

---

## 🖥️ APP FEATURES

| Feature                | Description                                    |
|------------------------|------------------------------------------------|
| Image Upload           | PNG/JPG breast ultrasound                      |
| GT Mask Upload         | Optional — enables metrics computation        |
| Segmentation Overlay   | Pink tumour region with contour outline        |
| Grad-CAM / Grad-CAM++  | Selectable via sidebar                         |
| Attention Rollout      | Multi-layer ViT attention visualization        |
| Per-Head Attention     | 12 individual attention heads (toggle)         |
| Probability Map        | Continuous segmentation probability heatmap   |
| Confidence Bars        | Class probability visualization                |
| Radar Chart            | Multi-metric performance visualization         |
| Risk Indicator         | Real-time malignancy risk bar                  |
| Download Buttons       | Export all XAI images as PNG                   |
| CLAHE Enhancement      | Pre-processing toggle                          |
| Threshold Slider       | Adjustable segmentation threshold              |

---

## 🔧 CONFIGURATION (config.py)

Key settings you may want to change:
```python
IMG_SIZE  = 224    # ViT input size (do not change without retraining)
BATCH_SIZE = 8     # Reduce to 4 if GPU memory < 8GB
EPOCHS    = 50     # Reduce to 20 for quick testing
LR        = 1e-4   # Learning rate
DATASET_PATH = "./data/BUS_dataset"   # Update if different path
```

---

## 🚀 QUICK START (After Training)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start app
streamlit run app.py

# 3. Open browser: http://localhost:8501

# 4. Upload a breast ultrasound PNG/JPG

# 5. View: segmentation + prediction + heatmaps instantly
```

---

## 📚 REFERENCES

1. Dosovitskiy et al. (2021) — "An Image is Worth 16×16 Words: ViT" — arXiv:2010.11929  
2. Ronneberger et al. (2015) — "U-Net: CNNs for Biomedical Segmentation"  
3. Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from Deep Networks"  
4. Abnar & Zuidema (2020) — "Quantifying Attention Flow in Transformers" — arXiv:2005.00928  
5. Al-Dhabyani et al. (2020) — BUSI Dataset — Data in Brief, DOI:10.1016/j.dib.2019.104863  
