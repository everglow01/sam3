# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Text-prompted image segmentation with SAM 3.

Usage:
    python text_segment.py --image path/to/img.jpg --text "yellow school bus"
    python text_segment.py --image img.jpg --text "person" --version sam3.1 --thresh 0.5
"""
import argparse

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--text", required=True, help="text prompt, e.g. 'a red car'")
    ap.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"],
                    help="HF checkpoint version (only used if --checkpoint missing)")
    ap.add_argument("--checkpoint", default="weights/sam3.1/sam3.1_multiplex.pt",
                    help="local .pt path; skips HF download entirely")
    ap.add_argument("--thresh", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--out", default="seg_out.png", help="output overlay path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer an explicit local checkpoint; otherwise resolve from HF.
    # For sam3.1 we hand the multiplex checkpoint to the image model explicitly;
    # the default (sam3) downloads its own image checkpoint.
    if args.checkpoint:
        ckpt = args.checkpoint
    elif args.version == "sam3.1":
        ckpt = download_ckpt_from_hf(version="sam3.1")
    else:
        ckpt = None
    model = build_sam3_image_model(device=device, checkpoint_path=ckpt)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.thresh)

    image = Image.open(args.image).convert("RGB")
    # SAM 3 runs inference in bfloat16 autocast (see official example notebooks).
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast("cpu", dtype=torch.bfloat16)
    with autocast:
        state = processor.set_image(image)
        out = processor.set_text_prompt(prompt=args.text, state=state)

    masks = out["masks"].cpu().numpy()              # [N, 1, H, W] bool
    boxes = out["boxes"].float().cpu().numpy()      # [N, 4] xyxy
    scores = out["scores"].float().cpu().numpy()    # [N]
    print(f"Found {len(scores)} instance(s) for prompt '{args.text}'")
    for i, s in enumerate(scores):
        x0, y0, x1, y1 = boxes[i]
        print(f"  #{i}: score={s:.3f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    save_overlay(np.array(image), masks, boxes, scores, args.out)
    print(f"Saved overlay -> {args.out}")


def save_overlay(image, masks, boxes, scores, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    rng = np.random.default_rng(0)
    for i in range(len(scores)):
        m = masks[i, 0]
        color = rng.random(3)
        overlay = np.zeros((*m.shape, 4))
        overlay[m] = [*color, 0.5]
        ax.imshow(overlay)
        x0, y0, x1, y1 = boxes[i]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, y0, f"{scores[i]:.2f}", color="white", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.7, pad=1))
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
