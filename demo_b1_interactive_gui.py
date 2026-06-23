# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
B1 GUI — interactive instance segmentation with the mouse.

Controls (in the window):
  Left click      add a foreground point  (green)
  Right click     add a background point  (red)
  Left drag       draw a box prompt
  r               reset all prompts
  s               save current view to demo_b1_gui.png
  q / Esc         quit

Usage:
  python demo_b1_interactive_gui.py --image img.jpg
"""
import argparse
import math
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# B1 (SAM 1 interactive task) needs the ORIGINAL sam3 checkpoint. The
# sam3.1_multiplex.pt tracker is the Object-Multiplex video tracker, whose mask
# decoder tokens are 16x larger and do NOT fit the image interactive head.
CKPT = "weights/sam3/sam3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = "SAM3 B1 - click to segment"
MAX_W = 1280  # cap display width; clicks are mapped back to original pixels


def verify_interactive_head(model, ckpt_path):
    """Fail fast if the checkpoint has no compatible SAM 1 interactive head."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    mapped = {k.replace("tracker.", "inst_interactive_predictor.model."): v
              for k, v in ckpt.items() if "tracker" in k}
    exp = model.state_dict()
    n_ok = sum(1 for k, v in mapped.items()
               if k in exp and exp[k].shape == v.shape)
    if n_ok == 0:
        sys.exit(
            f"\n[ERROR] '{ckpt_path}' has no compatible SAM 1 interactive head "
            "(it looks like the SAM 3.1 multiplex video tracker).\n"
            "B1 interactive segmentation needs the original 'sam3.pt' checkpoint "
            "from https://huggingface.co/facebook/sam3 .\n"
            "Pass it with --checkpoint weights/sam3/sam3.pt\n"
            "(A1/A2/A3 concept tasks still work fine with sam3.1_multiplex.pt.)")


def overlay_mask(bgr, mask, color=(0, 200, 255), alpha=0.5):
    m = np.asarray(mask).astype(bool)
    if m.ndim == 3:
        m = m[0]
    out = bgr.copy()
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=CKPT, help="original sam3.pt path")
    ap.add_argument("--out", default="demo_b1_gui.png")
    args = ap.parse_args()

    pil = Image.open(args.image).convert("RGB")
    base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = base.shape[:2]
    scale = min(1.0, MAX_W / W)
    disp_size = (int(W * scale), int(H * scale))

    print("Loading model + encoding image (one-time, ~30s)...")
    model = build_sam3_image_model(device=DEVICE, checkpoint_path=args.checkpoint,
                                   enable_inst_interactivity=True)
    verify_interactive_head(model, args.checkpoint)
    proc = Sam3Processor(model, device=DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = proc.set_image(pil)
    print("Ready. Left=fg point, Right=bg point, drag=box, r=reset, s=save, q=quit")

    st = {"points": [], "labels": [], "box": None,
          "down": None, "cur": None, "dragging": False, "dirty": False}

    def to_orig(x, y):
        return [int(x / scale), int(y / scale)]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["down"] = (x, y); st["dragging"] = True; st["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["dragging"]:
            st["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["dragging"]:
            st["dragging"] = False
            x0, y0 = st["down"]
            if math.hypot(x - x0, y - y0) < 5:           # click -> fg point
                st["points"].append(to_orig(x, y)); st["labels"].append(1)
            else:                                         # drag -> box
                ox0, oy0 = to_orig(x0, y0); ox1, oy1 = to_orig(x, y)
                st["box"] = [min(ox0, ox1), min(oy0, oy1),
                             max(ox0, ox1), max(oy0, oy1)]
            st["down"] = None; st["dirty"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:             # bg point
            st["points"].append(to_orig(x, y)); st["labels"].append(0)
            st["dirty"] = True

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    rendered = base.copy()

    def run_inference():
        pc = np.array(st["points"], dtype=float) if st["points"] else None
        pl = np.array(st["labels"]) if st["labels"] else None
        box = np.array(st["box"], dtype=float) if st["box"] else None
        if pc is None and box is None:
            return base.copy()
        multimask = box is None and len(st["points"]) <= 1
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            masks, scores, _ = model.predict_inst(
                state, point_coords=pc, point_labels=pl, box=box,
                multimask_output=multimask)
        scores = np.asarray(scores).ravel()
        best = int(np.argmax(scores))
        img = overlay_mask(base, masks[best])
        cv2.putText(img, f"score={scores[best]:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    while True:
        if st["dirty"]:
            rendered = run_inference()
            st["dirty"] = False
        view = cv2.resize(rendered, disp_size)
        # draw points
        for (px, py), lb in zip(st["points"], st["labels"]):
            cv2.circle(view, (int(px * scale), int(py * scale)), 6,
                       (0, 255, 0) if lb == 1 else (0, 0, 255), -1)
        # draw committed box
        if st["box"]:
            x0, y0, x1, y1 = st["box"]
            cv2.rectangle(view, (int(x0 * scale), int(y0 * scale)),
                          (int(x1 * scale), int(y1 * scale)), (0, 255, 0), 2)
        # live drag rectangle
        if st["dragging"] and st["down"] and st["cur"]:
            cv2.rectangle(view, st["down"], st["cur"], (0, 255, 255), 1)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            st.update(points=[], labels=[], box=None, dirty=False)
            rendered = base.copy()
        elif key == ord("s"):
            cv2.imwrite(args.out, rendered)
            print(f"Saved -> {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
