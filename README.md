# OncoVision — ViT-UNet for Breast Cancer Screening

End-to-end research-grade decision-support pipeline:

- **Vision Transformer encoder** + **U-Net style decoder** (TransUNet-inspired)
- **Multi-task output**: lesion **segmentation mask** + **Benign / Malignant** classification
- **Explainable AI**: Attention Rollout, Grad-CAM, Vanilla Saliency
- **FastAPI** backend serving model + dashboard
- **Clinical-tech dashboard** with overlay viewer, ECG-style "confidence signature", lesion morphometry, session case log

> ⚠️ **Research-grade tool.** This is a decision-support prototype — outputs are *not* a medical diagnosis. A qualified radiologist must perform final interpretation.

---

## Project layout

```
breast_cancer_vit/
├── model/
│   ├── vit_unet.py         # ViT encoder + U-Net decoder + multi-task head
│   ├── dataset.py          # BUSI-style dataset + synthetic fallback
│   └── train.py            # Training loop with combined Dice/BCE/CE loss
├── xai/
│   └── explainer.py        # Attention rollout, Grad-CAM, saliency
├── backend/
│   └── app.py              # FastAPI server (predict + dashboard host)
├── frontend/
│   ├── templates/index.html
│   └── static/{style.css, app.js}
├── checkpoints/            # Saved models go here
├── data/                   # Datasets go here (see "Data" below)
├── requirements.txt
├── run_smoke_test.py       # Verify everything works end-to-end
└── README.md
```

---

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU acceleration install the CUDA-matched PyTorch wheel from
[pytorch.org](https://pytorch.org/get-started/locally/).

## 2. Smoke test

Verify the whole stack runs without any data or checkpoint:

```bash
python run_smoke_test.py
```

You should see model creation, a forward+backward pass, the three XAI
methods, and overlay generation all succeed.

## 3. Data

Download a public dataset such as:

- **BUSI** — Breast Ultrasound Images Dataset
- **CBIS-DDSM** — mammography (after conversion)

Arrange it as:

```
data/BUSI/
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
└── normal/                 # optional — treated as benign with empty mask
```

If the folder is missing, the dataset class transparently switches to a
**synthetic on-the-fly dataset** so the pipeline still runs end-to-end.

## 4. Train

```bash
python -m model.train \
    --data_root data/BUSI \
    --epochs 30 \
    --batch_size 8 \
    --embed_dim 384 \
    --depth 6 \
    --num_heads 6 \
    --amp                # if CUDA available
```

The best model is saved to `checkpoints/best.pt`.

| Flag           | Default       | Meaning                         |
|----------------|---------------|---------------------------------|
| `--embed_dim`  | 384           | Transformer hidden dim          |
| `--depth`      | 6             | Number of transformer blocks    |
| `--num_heads`  | 6             | Number of attention heads       |
| `--lr`         | 1e-4          | AdamW learning rate             |
| `--val_split`  | 0.2           | Fraction reserved for validation|
| `--amp`        | false         | Enable mixed precision (CUDA)   |

For a full ViT-Base configuration (≈85 M params) use
`--embed_dim 768 --depth 12 --num_heads 12`.

## 5. Run the dashboard

```bash
# uses checkpoints/best.pt by default
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# or specify a different checkpoint
VITUNET_CHECKPOINT=checkpoints/last.pt uvicorn backend.app:app --port 8000
```

Open **http://localhost:8000** in a browser.

If no checkpoint exists, the server starts in **demo mode** with
randomly initialized weights — overlays render but predictions are
not meaningful. This is purely so you can verify UI plumbing without
training.

---

## Architecture

```
                       ┌──────────── Vision Transformer Encoder ───────────┐
                       │                                                    │
   image (B,3,H,W) ──▶ Patch Embed ─▶ +CLS +PosEmbed ─▶ N × Transformer blk │
                       │                                       │            │
                       └────────────────────┬──────────────────┴────────────┘
                                            │ tokens (B, N+1, D)
                                            │
                            ┌───────────────┴────────────────┐
                            │                                │
                       CLS token                    patch tokens reshape
                            │                                │
                            ▼                                ▼
                  Classification Head           CNN Decoder (4× upsample blocks)
                  (LayerNorm + MLP + 2)          ─────────────────────▶ seg_logits (B,1,H,W)
                            │
                            ▼
                  cls_logits (B,2)
```

### XAI

| Method               | What it shows                                                             | Where computed                            |
|----------------------|---------------------------------------------------------------------------|-------------------------------------------|
| **Attention Rollout**| How information from patches flows to the CLS token across all layers     | Multiplied attention matrices             |
| **Grad-CAM**         | Class-discriminative spatial map                                          | Hooks on `decoder.project` (1×1 conv)     |
| **Saliency**         | `∂score/∂input` magnitude — sanity baseline                               | One backward pass through input           |

---

## Dashboard features

- **Triple-output viewer** — switch instantly between segmentation overlay, attention rollout, Grad-CAM, and saliency, with live opacity slider and crosshair coordinate readout.
- **Confidence Signature (ECG)** — a unique waveform whose shape encodes the predicted class (sharp QRS-like spike for Malignant, smooth bump for Benign) and whose amplitude scales with model confidence.
- **Lesion Morphometry** — area, centroid, compactness, irregularity computed from the predicted mask.
- **Session Case Log** — every analyzed image is captured with thumbnail, time, class, and confidence; click to re-load.
- **Explainability Narrative** — the backend writes a short natural-language rationale mentioning shape regularity and attention focus.
- **Demo mode banner** when no checkpoint is loaded.
- **Synthetic demo button** — generates a procedural ultrasound-like image client-side, so reviewers can try the system without uploading any data.
- **Distinct typography** — Fraunces (editorial display), JetBrains Mono (technical readouts).

---

## API

`POST /api/predict`  multipart/form-data with field `file`

Returns:

```jsonc
{
  "request_id": "a1b2c3d4",
  "elapsed_ms": 142,
  "device": "cuda",
  "prediction": {
    "class_index": 1,
    "class_name": "Malignant",
    "probabilities": { "Benign": 0.182, "Malignant": 0.818 },
    "confidence": 0.818,
    "confidence_band": "Moderate"
  },
  "lesion": {
    "area_px": 1234, "area_pct": 2.46,
    "bbox": [x0, y0, x1, y1],
    "centroid": [122.4, 117.8],
    "compactness": 0.532,
    "irregularity": 0.41
  },
  "explanation": "The model predicts the lesion is **Malignant** ...",
  "images": {
    "original":            "data:image/png;base64,...",
    "segmentation":        "data:image/png;base64,...",
    "attention_rollout":   "data:image/png;base64,...",
    "gradcam":             "data:image/png;base64,...",
    "saliency":            "data:image/png;base64,..."
  }
}
```

Other endpoints: `GET /api/health`, `GET /api/model`.

---

## License & ethics

This codebase is for research and education. Real clinical deployment
requires regulatory clearance, prospective validation on the target
population, calibration, and integration with hospital systems.
