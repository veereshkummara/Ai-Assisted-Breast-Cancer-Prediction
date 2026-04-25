"""
FastAPI server.

    /                  -> dashboard HTML
    /static/...        -> dashboard assets
    POST /api/predict  -> upload image, returns JSON with base64 overlays
    GET  /api/health   -> simple health check
    GET  /api/model    -> model metadata

Run with:
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torchvision.transforms.functional as TF

# Local imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.vit_unet import ViTUNet  # noqa: E402
from model.dataset import IMAGENET_MEAN, IMAGENET_STD, denormalize  # noqa: E402
from xai.explainer import explain  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.environ.get("VITUNET_CHECKPOINT", str(ROOT / "checkpoints" / "best.pt"))
IMG_SIZE = 224

CLASS_NAMES = ["Benign", "Malignant"]
CLASS_COLORS = {
    0: (16, 185, 129),    # emerald — benign
    1: (236, 72, 153),    # pink — malignant
}


# ---------------------------------------------------------------------------
# Model loading (graceful fallback to randomly-initialised model)
# ---------------------------------------------------------------------------
def load_model():
    arch = dict(img_size=IMG_SIZE, embed_dim=384, depth=6, num_heads=6)
    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        if "args" in ckpt:
            a = ckpt["args"]
            arch.update(
                img_size=a.get("img_size", IMG_SIZE),
                embed_dim=a.get("embed_dim", 384),
                depth=a.get("depth", 6),
                num_heads=a.get("num_heads", 6),
            )
        model = ViTUNet(**arch).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=False)
        info = {"checkpoint": str(CHECKPOINT), "loaded": True, **arch}
        print(f"[backend] loaded checkpoint: {CHECKPOINT}")
    else:
        model = ViTUNet(**arch).to(DEVICE)
        info = {"checkpoint": None, "loaded": False, **arch}
        print(f"[backend] no checkpoint at {CHECKPOINT} — using randomly initialised "
              f"weights (DEMO MODE — predictions are not meaningful).")
    model.eval()
    return model, info


MODEL, MODEL_INFO = load_model()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.unsqueeze(0).to(DEVICE)


def heatmap_to_rgba(heat: np.ndarray, alpha=0.55) -> Image.Image:
    """Apply a perceptually-meaningful colormap (inferno-like) to a [0,1] map."""
    h = np.clip(heat, 0, 1)
    # 5-stop inferno-like colormap built without matplotlib dependency
    stops = np.array([
        [0,   0,   4],
        [40,  11,  84],
        [101, 21,  110],
        [159, 42,  99],
        [212, 72,  66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ], dtype=np.float32) / 255.0
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(h, pos, stops[:, c]) for c in range(3)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    a = (h * 255 * alpha).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    return Image.fromarray(rgba, mode="RGBA")


def overlay_heatmap(base_pil: Image.Image, heat: np.ndarray, alpha=0.55) -> Image.Image:
    base = base_pil.convert("RGBA")
    h_img = heatmap_to_rgba(heat, alpha=alpha).resize(base.size, Image.BILINEAR)
    return Image.alpha_composite(base, h_img)


def overlay_mask(base_pil: Image.Image, mask: np.ndarray, color=(236, 72, 153)) -> Image.Image:
    """Outline + translucent fill of binary mask onto image."""
    base = base_pil.convert("RGBA").copy()
    m = (mask.astype(np.uint8) * 255)
    fill_layer = Image.new("RGBA", base.size, color + (0,))
    fill_arr = np.array(fill_layer)
    m_resized = np.array(Image.fromarray(m).resize(base.size, Image.NEAREST))
    fill_arr[..., 3] = (m_resized > 127).astype(np.uint8) * 90
    fill_arr[..., 0:3] = np.array(color, dtype=np.uint8)
    fill_layer = Image.fromarray(fill_arr)
    out = Image.alpha_composite(base, fill_layer)

    # Outline
    edge_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)
    mask_pil = Image.fromarray(m_resized).convert("L")
    # crude edge: dilation - mask
    from PIL import ImageFilter
    edges = mask_pil.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges)
    ys, xs = np.where(edges_arr > 30)
    for y, x in zip(ys, xs):
        edge_draw.point((x, y), fill=color + (255,))
    out = Image.alpha_composite(out, edge_layer)
    return out


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pil_rgba_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Lesion measurements
# ---------------------------------------------------------------------------
def lesion_metrics(seg_mask: np.ndarray) -> dict:
    """Compute simple shape descriptors for the predicted lesion."""
    m = seg_mask.astype(bool)
    H, W = m.shape
    area = int(m.sum())
    if area == 0:
        return {
            "area_px": 0,
            "area_pct": 0.0,
            "bbox": None,
            "centroid": None,
            "compactness": None,
            "irregularity": None,
        }
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cy, cx = float(ys.mean()), float(xs.mean())

    # perimeter (4-connected edge count via discrete laplacian)
    pad = np.pad(m, 1, mode="constant")
    edges = (
        (pad[1:-1, 1:-1] & ~pad[:-2, 1:-1]).sum()
        + (pad[1:-1, 1:-1] & ~pad[2:, 1:-1]).sum()
        + (pad[1:-1, 1:-1] & ~pad[1:-1, :-2]).sum()
        + (pad[1:-1, 1:-1] & ~pad[1:-1, 2:]).sum()
    )
    perim = float(edges)
    compactness = (4 * np.pi * area) / (perim ** 2 + 1e-6)  # 1.0 = perfect circle
    bbox_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    irregularity = 1.0 - area / bbox_area  # 0 = fills bbox, higher = more irregular
    return {
        "area_px": area,
        "area_pct": round(area / (H * W) * 100, 2),
        "bbox": [x0, y0, x1, y1],
        "centroid": [round(cx, 1), round(cy, 1)],
        "compactness": round(float(compactness), 3),
        "irregularity": round(float(irregularity), 3),
    }


def confidence_band(p: float) -> str:
    if p >= 0.85:
        return "High"
    if p >= 0.65:
        return "Moderate"
    return "Low"


def build_explanation(pred_class: int, probs, metrics: dict) -> str:
    cls = CLASS_NAMES[pred_class]
    p = probs[pred_class]
    msg = (
        f"The model predicts the lesion is **{cls}** with {p*100:.1f}% confidence. "
    )
    if metrics["area_px"] == 0:
        msg += "No clear lesion was segmented — the prediction is based on diffuse tissue patterns."
        return msg

    if pred_class == 1:  # Malignant
        if metrics["compactness"] < 0.55:
            msg += "Attention is concentrated on an irregular, non-circular region — a feature commonly associated with malignant lesions. "
        if metrics["irregularity"] > 0.45:
            msg += "The segmented shape has spiculated or uneven margins. "
    else:  # Benign
        if metrics["compactness"] > 0.7:
            msg += "The segmented region is smooth and roughly circular — a pattern consistent with benign findings. "
    msg += (
        f"Lesion occupies ≈{metrics['area_pct']}% of the image. "
        "Heatmaps reveal which pixels most influenced the decision."
    )
    return msg


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="OncoVision — ViT-UNet Breast Cancer Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(ROOT / "frontend" / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = ROOT / "frontend" / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_INFO}


@app.get("/api/model")
def model_info():
    n_params = sum(p.numel() for p in MODEL.parameters())
    return {
        **MODEL_INFO,
        "n_parameters": n_params,
        "n_parameters_human": f"{n_params/1e6:.2f}M",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    t0 = time.time()
    img_resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = preprocess_image(img)

    res = explain(MODEL, x)
    pred_class = res["prediction"]
    probs = res["probs"]

    metrics = lesion_metrics(res["seg_mask"])
    metrics["confidence_band"] = confidence_band(probs[pred_class])
    explanation_text = build_explanation(pred_class, probs, metrics)

    # Build visual overlays
    color = CLASS_COLORS[pred_class]
    seg_overlay = overlay_mask(img_resized, res["seg_mask"], color=color)
    rollout_overlay = overlay_heatmap(img_resized, res["rollout"], alpha=0.55)
    gradcam_overlay = overlay_heatmap(img_resized, res["gradcam"], alpha=0.55)
    saliency_overlay = overlay_heatmap(img_resized, res["saliency"], alpha=0.55)

    payload = {
        "request_id": str(uuid.uuid4())[:8],
        "elapsed_ms": int((time.time() - t0) * 1000),
        "device": str(DEVICE),
        "prediction": {
            "class_index": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "probabilities": {
                CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "confidence": round(float(probs[pred_class]), 4),
            "confidence_band": metrics["confidence_band"],
        },
        "lesion": metrics,
        "explanation": explanation_text,
        "images": {
            "original": pil_to_b64(img_resized),
            "segmentation": pil_rgba_to_b64(seg_overlay),
            "attention_rollout": pil_rgba_to_b64(rollout_overlay),
            "gradcam": pil_rgba_to_b64(gradcam_overlay),
            "saliency": pil_rgba_to_b64(saliency_overlay),
        },
    }
    return JSONResponse(payload)


# Convenience: run with `python -m backend.app`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
