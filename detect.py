#!/usr/bin/env python3
"""
Helmet detection viewer (YOLO + OpenCV)

Green  = helmet
Red    = no helmet
Any other class the model outputs is ignored.

Usage:
    python detect.py video.mp4
    python detect.py 0                      # webcam
    python detect.py video.mp4 --track      # count unique violators
    python detect.py video.mp4 --save       # write annotated video
    python detect.py video.mp4 --skip 2 --show-conf

Keys: q / ESC = quit | SPACE = pause | s = save screenshot
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_MODEL = (
    "/home/Documents/projects/dresshelmet/helmet_detection_output/"
    "final_results/weights_folder/weights/best.pt"
)

# Names are normalised (lowercase, spaces -> underscores) before matching
HELMET_NAMES = {"helmet", "helmets", "with_helmet"}
NO_HELMET_NAMES = {"no_helmet", "no_helmets", "without_helmet"}

GREEN = (0, 220, 0)    # BGR
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

HELMET, NO_HELMET = "helmet", "no_helmet"
STYLE = {
    HELMET: {"color": GREEN, "label": "HELMET"},
    NO_HELMET: {"color": RED, "label": "NO HELMET"},
}
# ─────────────────────────────────────────────


def normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def build_class_map(names: dict) -> dict:
    """Map model class id -> HELMET / NO_HELMET. Unknown classes are dropped."""
    class_map = {}
    for cid, name in names.items():
        n = normalise(name)
        if n in HELMET_NAMES:
            class_map[cid] = HELMET
        elif n in NO_HELMET_NAMES:
            class_map[cid] = NO_HELMET
    return class_map


def parse_results(result, class_map: dict) -> list:
    """Return a list of (x1, y1, x2, y2, category, conf, track_id)."""
    dets = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = (
        boxes.id.cpu().numpy().astype(int)
        if boxes.id is not None
        else [None] * len(cls)
    )

    for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
        category = class_map.get(int(c))
        if category is None:
            continue  # not helmet / no-helmet -> ignore
        dets.append((x1, y1, x2, y2, category, float(p), tid))
    return dets


# ───────────────────────── drawing ─────────────────────────
def draw_box(frame, det, show_conf: bool):
    x1, y1, x2, y2, category, conf, tid = det
    color = STYLE[category]["color"]
    h, w = frame.shape[:2]
    base = min(w, h)

    scale = max(0.45, base / 1500)
    thick = max(1, round(base / 450))

    # thin full box + bold corner brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    L = max(8, int(min(x2 - x1, y2 - y1) * 0.22))
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, thick + 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, thick + 1, cv2.LINE_AA)

    # label pill
    label = STYLE[category]["label"]
    if tid is not None:
        label += f" #{tid}"
    if show_conf:
        label += f" {conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base_line = cv2.getTextSize(label, font, scale, 1)
    pad = 4
    ly2 = y1 if y1 - th - 2 * pad >= 0 else y1 + th + 2 * pad
    ly1 = ly2 - th - 2 * pad
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 2 * pad, ly2), color, -1)
    cv2.putText(frame, label, (x1 + pad, ly2 - pad), font, scale, BLACK, 1, cv2.LINE_AA)


def draw_hud(frame, helmets, no_helmets, fps, frame_idx, total, violators):
    h, w = frame.shape[:2]
    bar_h = max(36, h // 18)
    scale = bar_h / 52

    # translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = int(bar_h * 0.68)
    x = 12

    def put(text, color):
        nonlocal x
        cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, font, scale, 2)
        x += tw + 24

    put(f"HELMET {helmets}", GREEN)
    put(f"NO HELMET {no_helmets}", RED if no_helmets else WHITE)
    if violators is not None:
        put(f"VIOLATORS {violators}", RED if violators else WHITE)

    right = f"{fps:4.1f} FPS"
    if total > 0:
        right += f"  |  {frame_idx}/{total}"
    (tw, _), _ = cv2.getTextSize(right, font, scale * 0.8, 1)
    cv2.putText(frame, right, (w - tw - 12, y), font, scale * 0.8, WHITE, 1, cv2.LINE_AA)

    # red frame border when someone has no helmet
    if no_helmets:
        t = max(2, w // 300)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, t)


# ───────────────────────── main ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Helmet / no-helmet detection on video or webcam")
    p.add_argument("source", help="video file path, or webcam index like 0")
    p.add_argument("--model", default=DEFAULT_MODEL, help="path to YOLO weights")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--device", default=None, help="cpu, 0, 1, ... (default: auto)")
    p.add_argument("--skip", type=int, default=1, help="run inference every Nth frame (speed boost)")
    p.add_argument("--show-conf", action="store_true", help="show confidence scores")
    p.add_argument("--track", action="store_true", help="enable tracking + unique violator count")
    p.add_argument("--save", nargs="?", const="auto", default=None, metavar="OUT.mp4",
                   help="save annotated video (optional output path)")
    p.add_argument("--no-display", action="store_true", help="headless mode (no window)")
    return p.parse_args()


def main():
    args = parse_args()

    # source: webcam index or file
    is_webcam = args.source.isdigit()
    if is_webcam:
        source = int(args.source)
    else:
        source = args.source
        if not Path(source).exists():
            sys.exit(f"[ERROR] Video not found: {source}")

    if not Path(args.model).exists():
        sys.exit(f"[ERROR] Model not found: {args.model}")

    print("[INFO] Loading model...")
    model = YOLO(args.model)
    class_map = build_class_map(model.names)
    print(f"[INFO] Model classes : {model.names}")
    print(f"[INFO] Using classes : { {model.names[k]: v for k, v in class_map.items()} }")
    if HELMET not in class_map.values() or NO_HELMET not in class_map.values():
        print("[WARN] Could not match both helmet and no-helmet classes. "
              "Update HELMET_NAMES / NO_HELMET_NAMES at the top of the script.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open video source.")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    total = 0 if is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {width}x{height} @ {fps_src:.1f} fps"
          + (f" | {total} frames" if total else " | live"))

    # optional writer
    writer = None
    if args.save:
        out_path = (
            f"{Path(args.source).stem}_helmet.mp4" if args.save == "auto" and not is_webcam
            else ("webcam_helmet.mp4" if args.save == "auto" else args.save)
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (width, height))
        print(f"[INFO] Saving to: {out_path}")

    show = not args.no_display
    if show:
        print("[INFO] q/ESC = quit | SPACE = pause | s = screenshot")

    frame_delay = 1000.0 / fps_src
    loop_times = deque(maxlen=30)
    dets = []
    violator_ids = set()
    frame_idx = 0
    shot_idx = 0
    skip = max(1, args.skip)
    infer_kwargs = dict(conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0 = time.perf_counter()

        # inference (reuse last detections on skipped frames)
        if (frame_idx - 1) % skip == 0:
            if args.track:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", **infer_kwargs)[0]
            else:
                result = model.predict(frame, **infer_kwargs)[0]
            dets = parse_results(result, class_map)
            for d in dets:
                if d[4] == NO_HELMET and d[6] is not None:
                    violator_ids.add(int(d[6]))

        helmets = sum(1 for d in dets if d[4] == HELMET)
        no_helmets = sum(1 for d in dets if d[4] == NO_HELMET)

        for d in dets:
            draw_box(frame, d, args.show_conf)

        fps = (len(loop_times) / sum(loop_times)) if loop_times else 0.0
        draw_hud(frame, helmets, no_helmets, fps, frame_idx, total,
                 len(violator_ids) if args.track else None)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Helmet Detection", frame)
            elapsed = (time.perf_counter() - t0) * 1000.0
            wait = 1 if is_webcam else max(1, int(round(frame_delay - elapsed)))
            key = cv2.waitKey(wait) & 0xFF

            if key in (ord("q"), 27):
                print("[INFO] Stopped by user.")
                break
            elif key == ord("s"):
                shot_idx += 1
                name = f"screenshot_{shot_idx:03d}.png"
                cv2.imwrite(name, frame)
                print(f"[INFO] Saved {name}")
            elif key == ord(" "):
                while True:  # paused
                    k = cv2.waitKey(50) & 0xFF
                    if k == ord(" "):
                        break
                    if k in (ord("q"), 27):
                        frame_idx = total + 1  # force outer exit
                        break
                if frame_idx > total > 0:
                    break

        loop_times.append(time.perf_counter() - t0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"[DONE] {frame_idx} frames processed.")
    if args.track:
        print(f"[DONE] Unique no-helmet violators: {len(violator_ids)}")


if __name__ == "__main__":
    main()
