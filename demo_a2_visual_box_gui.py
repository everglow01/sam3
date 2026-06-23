# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
A2 GUI — visual box-exemplar segmentation with the mouse.

Drag a box around ONE object as an example; SAM 3 segments every object of the
same concept. Left drag = positive example, right drag = negative (exclude).

Controls (in the window):
  Left drag       positive exemplar box  (green)
  Right drag      negative exemplar box  (red, excludes that concept)
  r               reset all boxes
  s               save current view to demo_a2_gui.png
  q / Esc         quit

Usage:
  python demo_a2_visual_box_gui.py --image img.jpg
  python demo_a2_visual_box_gui.py --image img.jpg --text car   # combine with text
"""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CKPT = "weights/sam3.1/sam3.1_multiplex.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = "SAM3 A2 - drag a box to find all same-class objects"
MAX_W = 1280
RNG = np.random.default_rng(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default="", help="optional text to combine with boxes")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="demo_a2_gui.png")
    args = ap.parse_args()

    pil = Image.open(args.image).convert("RGB")
    base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = base.shape[:2]
    scale = min(1.0, MAX_W / W)
    disp_size = (int(W * scale), int(H * scale))

    print("Loading model + encoding image (one-time, ~30s)...")
    model = build_sam3_image_model(device=DEVICE, checkpoint_path=CKPT)
    proc = Sam3Processor(model, device=DEVICE, confidence_threshold=args.thresh)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = proc.set_image(pil)
    print("Ready. Left drag=positive box, Right drag=negative box, r=reset, s=save, q=quit")

    # boxes: list of (xyxy_pixels, label_bool)
    st = {"boxes": [], "down": None, "cur": None, "btn": None, "dirty": False}

    def to_orig(x, y):
        return [int(x / scale), int(y / scale)]

    def on_mouse(event, x, y, flags, _):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            st["down"] = (x, y); st["cur"] = (x, y)
            st["btn"] = "L" if event == cv2.EVENT_LBUTTONDOWN else "R"
        elif event == cv2.EVENT_MOUSEMOVE and st["down"]:
            st["cur"] = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and st["down"]:
            x0, y0 = st["down"]
            ox0, oy0 = to_orig(x0, y0); ox1, oy1 = to_orig(x, y)
            if abs(ox1 - ox0) > 3 and abs(oy1 - oy0) > 3:
                box = [min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1)]
                st["boxes"].append((box, st["btn"] == "L"))
                st["dirty"] = True
            st["down"] = None; st["btn"] = None

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    rendered = base.copy()

    def run_inference():
        proc.reset_all_prompts(state)          # keeps cached image features
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            if args.text:
                proc.set_text_prompt(prompt=args.text, state=state)
            out = state
            for box, label in st["boxes"]:
                x0, y0, x1, y1 = box
                norm = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                        (x1 - x0) / W, (y1 - y0) / H]
                out = proc.add_geometric_prompt(box=norm, label=label, state=state)
        if "masks" not in out:
            return base.copy()
        masks = out["masks"].cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        img = base.copy()
        for i, m in enumerate(masks):
            mm = np.asarray(m).astype(bool)
            if mm.ndim == 3:
                mm = mm[0]
            color = (RNG.random(3) * 255).astype(int)
            img[mm] = (img[mm] * 0.5 + color * 0.5).astype(np.uint8)
        cv2.putText(img, f"{len(scores)} matches", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    while True:
        if st["dirty"]:
            rendered = run_inference()
            st["dirty"] = False
        view = cv2.resize(rendered, disp_size)
        for box, label in st["boxes"]:
            x0, y0, x1, y1 = box
            cv2.rectangle(view, (int(x0 * scale), int(y0 * scale)),
                          (int(x1 * scale), int(y1 * scale)),
                          (0, 255, 0) if label else (0, 0, 255), 2)
        if st["down"] and st["cur"]:
            c = (0, 255, 0) if st["btn"] == "L" else (0, 0, 255)
            cv2.rectangle(view, st["down"], st["cur"], c, 1)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            st.update(boxes=[], down=None, btn=None, dirty=False)
            rendered = base.copy()
        elif key == ord("s"):
            cv2.imwrite(args.out, rendered)
            print(f"Saved -> {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
