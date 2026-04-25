"""
ViT-UNet Hybrid Model for Breast Cancer Detection
==================================================
Vision Transformer encoder + U-Net style decoder.
Outputs:
    - Pixel-wise segmentation mask (lesion location)
    - Binary classification (Malignant / Benign)
    - Attention maps (used by XAI module)

Architecture inspired by TransUNet (Chen et al., 2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vision Transformer Building Blocks
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Splits image into patches and linearly projects them to embedding space."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/p, W/p) -> (B, n_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention that also returns attention weights for XAI."""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn  # attn: (B, heads, N, N)
        return out, None


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attn_out, attn_w = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class VisionTransformerEncoder(nn.Module):
    """ViT encoder that stores attention maps from every layer for XAI use."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_all_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_maps = []
        for block in self.blocks:
            x, a = block(x, return_attn=return_all_attn)
            if return_all_attn:
                attn_maps.append(a)

        x = self.norm(x)
        return x, attn_maps  # x: (B, N+1, D)


# ---------------------------------------------------------------------------
# CNN Decoder (U-Net Style)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Upsamples by 2x and refines with two Conv-BN-ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """
    Upsamples ViT feature map (14x14 -> 224x224) producing a single-channel
    segmentation logit map for the lesion.
    """

    def __init__(self, embed_dim=768, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.project = nn.Conv2d(embed_dim, decoder_channels[0], 1)
        self.up1 = DecoderBlock(decoder_channels[0], decoder_channels[1])  # 14->28
        self.up2 = DecoderBlock(decoder_channels[1], decoder_channels[2])  # 28->56
        self.up3 = DecoderBlock(decoder_channels[2], decoder_channels[3])  # 56->112
        self.up4 = DecoderBlock(decoder_channels[3], decoder_channels[3])  # 112->224
        self.head = nn.Conv2d(decoder_channels[3], 1, 1)

    def forward(self, feat_map):
        x = self.project(feat_map)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)  # (B, 1, 224, 224)


# ---------------------------------------------------------------------------
# Full Multi-Task Model
# ---------------------------------------------------------------------------
class ViTUNet(nn.Module):
    """
    Multi-task model:
        - segmentation_logits: (B, 1, H, W)
        - classification_logits: (B, 2)   -> [benign, malignant]
        - attention maps from every transformer layer (for XAI)
    """

    CLASS_NAMES = ["Benign", "Malignant"]

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = SegmentationDecoder(embed_dim=embed_dim)

        # Classification head from CLS token
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def _tokens_to_feature_map(self, tokens):
        """
        tokens: (B, N+1, D) -> drop CLS, reshape to (B, D, H/p, W/p)
        """
        B, N1, D = tokens.shape
        patches = tokens[:, 1:, :]  # drop CLS
        gs = self.encoder.grid_size
        feat = patches.transpose(1, 2).reshape(B, D, gs, gs)
        return feat

    def forward(self, x, return_attn=False):
        tokens, attn_maps = self.encoder(x, return_all_attn=return_attn)
        cls_tok = tokens[:, 0]
        feat_map = self._tokens_to_feature_map(tokens)

        seg_logits = self.decoder(feat_map)
        cls_logits = self.cls_head(cls_tok)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "attn_maps": attn_maps,
            "feat_map": feat_map,
            "tokens": tokens,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1).float()
        intersection = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class MultiTaskLoss(nn.Module):
    """Combined BCE + Dice (segmentation) + CrossEntropy (classification)."""

    def __init__(self, w_seg=1.0, w_cls=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.w_seg = w_seg
        self.w_cls = w_cls

    def forward(self, outputs, mask, label):
        seg_logits = outputs["seg_logits"]
        cls_logits = outputs["cls_logits"]
        loss_bce = self.bce(seg_logits, mask.float())
        loss_dice = self.dice(seg_logits, mask)
        loss_cls = self.ce(cls_logits, label)
        loss = self.w_seg * (loss_bce + loss_dice) + self.w_cls * loss_cls
        return loss, {
            "bce": loss_bce.item(),
            "dice": loss_dice.item(),
            "cls": loss_cls.item(),
            "total": loss.item(),
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = ViTUNet(img_size=224, depth=4, num_heads=8, embed_dim=384)  # tiny for test
    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_attn=True)
    print("seg_logits:", out["seg_logits"].shape)
    print("cls_logits:", out["cls_logits"].shape)
    print("num attn maps:", len(out["attn_maps"]))
    print("attn_map[0] shape:", out["attn_maps"][0].shape)

    # Loss test
    mask = torch.randint(0, 2, (2, 1, 224, 224))
    label = torch.randint(0, 2, (2,))
    crit = MultiTaskLoss()
    loss, parts = crit(out, mask, label)
    print("loss:", parts)
