"""
Explainable AI utilities for the ViT-UNet model.

Provides three complementary visualisations:

1. Attention Rollout  (Abnar & Zuidema, 2020)
   Aggregates attention from all transformer layers to show *where the
   transformer is looking* for the classification decision.

2. Grad-CAM on the segmentation feature map
   Class-discriminative localisation map computed from gradients flowing
   into the decoder's projected feature map.

3. Saliency
   Vanilla input-gradient saliency, used as a sanity-check baseline.

All methods return a (H, W) numpy array in [0, 1] so the backend can
turn it into a heatmap overlay.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Attention Rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def attention_rollout(attn_maps, discard_ratio: float = 0.0,
                      head_fusion: str = "mean") -> np.ndarray:
    """
    Args:
        attn_maps: list[Tensor], each (1, heads, N+1, N+1)
        discard_ratio: fraction of lowest attentions to zero out per layer
        head_fusion: "mean", "max", or "min"
    Returns:
        heatmap: (gs, gs) numpy array in [0, 1] — attention from CLS to patches.
    """
    assert len(attn_maps) > 0, "no attention maps provided"
    device = attn_maps[0].device
    n_tokens = attn_maps[0].shape[-1]
    result = torch.eye(n_tokens, device=device)

    for attn in attn_maps:
        if head_fusion == "mean":
            a = attn.mean(dim=1)
        elif head_fusion == "max":
            a = attn.max(dim=1).values
        elif head_fusion == "min":
            a = attn.min(dim=1).values
        else:
            raise ValueError(head_fusion)

        if discard_ratio > 0:
            flat = a.view(a.size(0), -1)
            k = int(flat.size(-1) * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, dim=-1, largest=False)
                flat.scatter_(-1, idx, 0)
            a = flat.view_as(a)

        # add residual + renormalise rows
        a = a + torch.eye(n_tokens, device=device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)

        result = a[0] @ result  # batch=1

    cls_to_patches = result[0, 1:]  # (N,)
    gs = int(cls_to_patches.numel() ** 0.5)
    heat = cls_to_patches.reshape(gs, gs).cpu().numpy()
    heat -= heat.min()
    heat /= heat.max() + 1e-8
    return heat


# ---------------------------------------------------------------------------
# 2. Grad-CAM on the ViT feature map
# ---------------------------------------------------------------------------
class ViTGradCAM:
    """
    Grad-CAM-style localisation against the spatial feature map produced
    by the encoder (the input to the segmentation decoder).
    Implemented with hooks on `model.decoder.project`.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        target_module = model.decoder.project
        self._h1 = target_module.register_forward_hook(self._save_activation)
        self._h2 = target_module.register_full_backward_hook(self._save_gradient)

    # hooks
    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Args:
            x: (1, 3, H, W) normalised input tensor on the same device as model.
            class_idx: 0 (Benign) or 1 (Malignant). Defaults to predicted class.
        Returns:
            cam: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        logits = out["cls_logits"]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # Channel-wise GAP of gradients -> weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_idx, F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    def remove(self):
        self._h1.remove()
        self._h2.remove()


# ---------------------------------------------------------------------------
# 3. Vanilla saliency
# ---------------------------------------------------------------------------
def input_saliency(model, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Absolute gradient of the predicted class score w.r.t. the input."""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    logits = out["cls_logits"]
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())
    score = logits[0, class_idx]
    score.backward()
    sal = x.grad.detach().abs().max(dim=1)[0].squeeze().cpu().numpy()
    sal -= sal.min()
    sal /= sal.max() + 1e-8
    return sal


# ---------------------------------------------------------------------------
# Helpers for the backend to assemble explanations
# ---------------------------------------------------------------------------
def upscale_heatmap(heat: np.ndarray, target_hw):
    """Bicubic upscale a (gs, gs) attention map to (H, W)."""
    t = torch.from_numpy(heat).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bicubic", align_corners=False)
    out = t.squeeze().numpy()
    out -= out.min()
    out /= out.max() + 1e-8
    return out


def explain(model, x: torch.Tensor):
    """
    Run a single sample through both XAI methods.

    Returns dict with:
        rollout: (H, W) attention rollout heatmap
        gradcam: (H, W) Grad-CAM heatmap
        saliency: (H, W) input saliency map
        prediction: int  (0 benign, 1 malignant)
        probs: (2,)
        seg_mask: (H, W) bool predicted segmentation
    """
    model.eval()
    H, W = x.shape[-2:]

    # 1. attention rollout (needs attn maps)
    with torch.no_grad():
        out = model(x, return_attn=True)
        seg_prob = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        seg_mask = seg_prob > 0.5
    rollout = attention_rollout(out["attn_maps"])
    rollout = upscale_heatmap(rollout, (H, W))

    # 2. grad-cam
    cam_module = ViTGradCAM(model)
    try:
        gradcam, pred_idx, probs = cam_module(x)
    finally:
        cam_module.remove()

    # 3. saliency
    sal = input_saliency(model, x, class_idx=pred_idx)

    return {
        "rollout": rollout,
        "gradcam": gradcam,
        "saliency": sal,
        "prediction": int(pred_idx),
        "probs": probs.tolist(),
        "seg_mask": seg_mask,
        "seg_prob": seg_prob,
    }
