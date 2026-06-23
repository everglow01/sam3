# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
A3 — Confidence-threshold sweep.

Run one text prompt at several confidence thresholds and compare how many
instances survive. Higher threshold = fewer but more confident detections.

Usage:
  python demo_a3_threshold.py --image img.jpg --text car
  python demo_a3_threshold.py --image img.jpg --text car --thresholds 0.2 0.5 0.8 0.95
"""
import argparse

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CKPT = "weights/sam3.1/sam3.1_multiplex.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)


def draw_panel(ax, image, masks, scores):
    ax.imshow(image)
    for i, m in enumerate(masks):
        m = np.asarray(m).astype(bool)
        if m.ndim == 3:
            m = m[0]
        color = RNG.random(3)
        ov = np.zeros((*m.shape, 4))
        ov[m] = [*color, 0.5]
        ax.imshow(ov)
        ys, xs = np.where(m)
        if len(xs):
            ax.text(xs.min(), ys.min(), f"{scores[i]:.2f}", color="white",
                    fontsize=8, bbox=dict(facecolor=color, alpha=0.7, pad=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--out", default="demo_a3.png")
    args = ap.parse_args()

    model = build_sam3_image_model(device=DEVICE, checkpoint_path=CKPT)
    proc = Sam3Processor(model, device=DEVICE, confidence_threshold=0.5)
    image = Image.open(args.image).convert("RGB")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(args.thresholds)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = proc.set_image(image)
        proc.set_text_prompt(prompt=args.text, state=state)
        for t, ax in zip(args.thresholds, axes):
            out = proc.set_confidence_threshold(t, state)
            masks = out["masks"].cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            print(f"A3 thresh={t}: {len(scores)} instance(s)")
            draw_panel(ax, np.array(image), masks, scores)
            ax.set_title(f"thresh={t}  ->  {len(scores)} '{args.text}'", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
