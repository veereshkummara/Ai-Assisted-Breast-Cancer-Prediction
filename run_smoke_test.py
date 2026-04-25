"""
Quick smoke-test that runs the whole stack on synthetic data:
    python run_smoke_test.py

Covers:
    1. building the model
    2. one forward + backward pass
    3. running the explainer
    4. exercising the lesion-metrics & overlay code from backend.app
No checkpoint or dataset is required.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from model.vit_unet import ViTUNet, MultiTaskLoss
from model.dataset import BreastCancerDataset
from xai.explainer import explain
from backend.app import (
    preprocess_image, lesion_metrics, overlay_mask, overlay_heatmap,
    build_explanation, CLASS_NAMES,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    # 1. Model
    model = ViTUNet(img_size=224, embed_dim=192, depth=3, num_heads=6).to(device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[smoke] model = {n:.2f}M params")

    # 2. Dataset (synthetic)
    ds = BreastCancerDataset(root="data/none", img_size=224, train=True)
    img, mask, label = ds[0]
    print(f"[smoke] sample shapes: img={tuple(img.shape)} mask={tuple(mask.shape)} label={label.item()}")

    # 3. Forward + backward
    x = img.unsqueeze(0).to(device)
    m = mask.unsqueeze(0).to(device)
    y = label.unsqueeze(0).to(device)
    out = model(x, return_attn=True)
    crit = MultiTaskLoss()
    loss, parts = crit(out, m, y)
    loss.backward()
    print(f"[smoke] loss = {parts['total']:.4f}  (cls={parts['cls']:.3f} dice={parts['dice']:.3f})")

    # 4. Explainer
    res = explain(model, x)
    print(f"[smoke] prediction = {CLASS_NAMES[res['prediction']]}  probs = {res['probs']}")
    print(f"[smoke] heatmap shapes: rollout={res['rollout'].shape} gradcam={res['gradcam'].shape}")

    # 5. Backend pipeline
    pil = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    xt = preprocess_image(pil)
    res2 = explain(model, xt)
    metrics = lesion_metrics(res2["seg_mask"])
    metrics["confidence_band"] = "Low"
    print(f"[smoke] metrics = {metrics}")
    text = build_explanation(res2["prediction"], res2["probs"], metrics)
    print(f"[smoke] explanation = {text[:120]}...")

    # 6. Overlays
    pil_resized = pil.convert("RGB").resize((224, 224))
    overlay_mask(pil_resized, res2["seg_mask"])
    overlay_heatmap(pil_resized, res2["rollout"])
    overlay_heatmap(pil_resized, res2["gradcam"])
    print("[smoke] overlays OK")

    print("[smoke] ✅ all passed")


if __name__ == "__main__":
    main()
