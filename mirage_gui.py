#!/usr/bin/env python3
"""Mirage GUI - a small local web UI for the Cactus Mirage vision model.

Unofficial project, not affiliated with Cactus Compute.
Everything runs locally; the page is only reachable from this machine.
"""
import argparse
import json
import os
import tempfile
import time

import numpy as np
import cactus
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter, ImageOps

model = cactus.Mirage()


# ---------- optional RapidOCR (better text engine) ----------
def _load_rapid():
    """Return (callable or None, error text). Tries the old and the new package."""
    errs = []
    try:
        from rapidocr_onnxruntime import RapidOCR
        eng = RapidOCR()

        def run(arr):
            res, _ = eng(arr)
            return list(res or [])

        return run, ""
    except Exception as e:  # noqa: BLE001
        errs.append(f"rapidocr_onnxruntime: {e!r}")
    try:
        from rapidocr import RapidOCR
        eng = RapidOCR()

        def run(arr):
            out = eng(arr)
            if out is None or getattr(out, "boxes", None) is None:
                return []
            return list(zip(out.boxes, out.txts, out.scores))

        return run, ""
    except Exception as e:  # noqa: BLE001
        errs.append(f"rapidocr: {e!r}")
    return None, " | ".join(errs)


RAPID, RAPID_ERR = _load_rapid()
if RAPID is None:
    print("RapidOCR unavailable:", RAPID_ERR)
else:
    print("RapidOCR ready")


# ---------- helpers ----------
def preprocess(img, gray, contrast, invert_dark, sharpen):
    img = img.convert("RGB")
    if invert_dark and np.asarray(img.convert("L")).mean() < 110:
        img = ImageOps.invert(img)
    if gray:
        img = ImageOps.grayscale(img).convert("RGB")
    if contrast:
        img = ImageOps.autocontrast(img)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    return img


def tile_boxes(W, H, n, overlap=0.15):
    if n == 1:
        return [(0, 0, W, H)]
    tw, th = W / n, H / n
    ox, oy = tw * overlap, th * overlap
    return [
        tuple(int(v) for v in (max(0, c * tw - ox), max(0, r * th - oy),
                               min(W, (c + 1) * tw + ox), min(H, (r + 1) * th + oy)))
        for r in range(n) for c in range(n)
    ]


def to_full(b, box, up, mode):
    """Map a box returned for a (possibly upscaled) crop back to full-image pixels."""
    cw, ch = box[2] - box[0], box[3] - box[1]
    if mode.startswith("0-1000"):
        sx, sy = cw / 1000, ch / 1000
    else:
        sx = sy = 1 / up
    return [box[0] + b[0] * sx, box[1] + b[1] * sy, box[0] + b[2] * sx, box[1] + b[3] * sy]


def crop_up(img, box, up):
    c = img.crop(box)
    if up != 1:
        c = c.resize((max(1, int(c.width * up)), max(1, int(c.height * up))), Image.LANCZOS)
    return c


def mirage_pass(img, box, up, mode):
    crop = crop_up(img, box, up)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        crop.save(path)
        rec = model.describe(path)
    finally:
        os.remove(path)
    if not isinstance(rec, dict):
        return {"raw": str(rec)}, [], []
    texts = [
        {"string": str(t.get("string", "")), "bbox": to_full(t["bbox"], box, up, mode),
         "score": float(t.get("score", 0)), "pri": 0}
        for t in rec.get("text", []) if t.get("bbox")
    ]
    objs = [
        {"name": o.get("name", "?"), "bbox": to_full(o["bbox"], box, up, mode),
         "score": float(o.get("score", 0)), "desc": o.get("description") or {}}
        for o in rec.get("objects", []) if o.get("bbox")
    ]
    return rec, texts, objs


def rapid_pass(img, box, up):
    crop = crop_up(img, box, up)
    out = []
    for b, t, s in RAPID(np.array(crop)):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        out.append({
            "string": str(t), "score": float(s), "pri": 1,
            "bbox": [box[0] + min(xs) / up, box[1] + min(ys) / up,
                     box[0] + max(xs) / up, box[1] + max(ys) / up],
        })
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def overlap_min(a, b):
    """Intersection area divided by the smaller box's area (catches fragments inside a bigger box)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / smaller if smaller > 0 else 0


def dedupe(items, same=lambda a, b: True, thr=0.4, contain=0.8):
    """Drop duplicates from overlapping tiles/engines. Preferred first: RapidOCR, longer text, higher score."""
    kept = []
    order = sorted(items, key=lambda t: (t.get("pri", 0), len(t.get("string", "").strip()), t["score"]),
                   reverse=True)
    for t in order:
        if not any(same(t, k) and (iou(t["bbox"], k["bbox"]) >= thr
                                   or overlap_min(t["bbox"], k["bbox"]) >= contain) for k in kept):
            kept.append(t)
    return kept


def to_lines(texts):
    """Group text boxes into reading-order lines."""
    if not texts:
        return ""
    hs = sorted(t["bbox"][3] - t["bbox"][1] for t in texts)
    med = hs[len(hs) // 2] or 10
    texts = sorted(texts, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)
    lines, cur, cur_y = [], [], 0
    for t in texts:
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        if cur and abs(cy - cur_y) > 0.6 * med:
            lines.append(cur)
            cur = []
        if not cur:
            cur_y = cy
        cur.append(t)
    if cur:
        lines.append(cur)
    return "\n".join(
        "  ".join(x["string"] for x in sorted(line, key=lambda t: t["bbox"][0])) for line in lines
    )


def annotate(img, texts, objs):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for o in objs:
        d.rectangle(o["bbox"], outline=(0, 200, 0), width=3)
        d.text((o["bbox"][0] + 3, o["bbox"][1] + 3), f"{o['name']} {o['score']:.2f}", fill=(0, 255, 0))
    for t in texts:
        d.rectangle(t["bbox"], outline=(255, 60, 60), width=2)
        d.text((t["bbox"][0], max(0, t["bbox"][1] - 11)), t["string"], fill=(255, 255, 0))
    return out


# ---------- main processing ----------
def run(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    if img is None:
        return None, "", "", None, None, "Upload or paste an image first."
    t0 = time.time()
    orig = img.convert("RGB")
    pre = preprocess(orig, gray, contrast, invert, sharpen)
    W, H = pre.size
    n = int(str(grid)[0])
    want_mirage = engine in ("Mirage", "Both") or RAPID is None  # fall back to Mirage text
    want_rapid = engine in ("RapidOCR", "Both") and RAPID is not None
    note = ""
    if engine != "Mirage" and RAPID is None:
        note = f" | RapidOCR unavailable ({RAPID_ERR or 'not installed'}), used Mirage only"

    # whole-image Mirage pass: scene + objects (+ text)
    rec0, t_whole, o_whole = mirage_pass(pre, (0, 0, W, H), 1, mode)
    texts = [t for t in t_whole if want_mirage and t["score"] >= min_mir]

    # tiled passes zoom into small text
    if want_mirage and n > 1:
        for box in tile_boxes(W, H, n):
            texts += [t for t in mirage_pass(pre, box, up, mode)[1] if t["score"] >= min_mir]

    # RapidOCR finds text regions itself, so it runs on the whole image (tiles would cut long lines)
    if want_rapid:
        eff = max(1.0, min(float(up), 3000.0 / max(W, H)))  # cap the upscaled size
        texts += [t for t in rapid_pass(pre, (0, 0, W, H), eff) if t["score"] >= min_rap]

    texts = [t for t in texts if len(t["string"].strip()) > 1 and any(c.isalnum() for c in t["string"])]
    texts = dedupe(texts)
    objs = dedupe(
        [o for o in o_whole if o["name"] not in ("text", "screen") and o["score"] >= min_obj],
        same=lambda a, b: a["name"] == b["name"],
    )

    scene = rec0.get("scene") if isinstance(rec0, dict) else None
    summary = []
    if scene:
        summary.append("SCENE: " + ", ".join(f"{k}: {v}" for k, v in scene.items()))
    for o in objs:
        desc = ", ".join(f"{k}={v}" for k, v in o["desc"].items())
        summary.append(f"- {o['name']} ({o['score']:.2f}) {desc}".rstrip())

    data = {
        "scene": scene,
        "objects": [{"name": o["name"], "score": round(o["score"], 3),
                     "bbox": [round(x) for x in o["bbox"]], "description": o["desc"]} for o in objs],
        "text": [{"string": t["string"], "score": round(t["score"], 3),
                  "bbox": [round(x) for x in t["bbox"]]} for t in texts],
    }
    fd, jpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    tiles_used = n * n if (want_mirage and n > 1) else 1
    status = (f"{len(texts)} text boxes, {len(objs)} objects | Mirage tiles: {tiles_used} | "
              f"coords: {mode} | {time.time() - t0:.1f}s{note}")
    return annotate(orig, texts, objs), to_lines(texts), "\n".join(summary), data, jpath, status


# ---------- UI ----------
def textbox(label, lines):
    try:
        return gr.Textbox(label=label, lines=lines, show_copy_button=True)
    except TypeError:  # newer Gradio versions renamed this option
        return gr.Textbox(label=label, lines=lines)


with gr.Blocks(title="Mirage GUI") as demo:
    gr.Markdown("## Mirage GUI")
    gr.Markdown("RapidOCR: **ready**" if RAPID else
                f"RapidOCR: **not available** ({RAPID_ERR or 'not installed'}). Mirage-only mode.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Image (upload or paste)", sources=["upload", "clipboard"])
            engine = gr.Radio(["Mirage", "RapidOCR", "Both"], value="Both" if RAPID else "Mirage",
                              label="Text engine")
            grid = gr.Radio(["1x1 (whole)", "2x2", "3x3"], value="2x2", label="Mirage tiles (zoom into small text)")
            up = gr.Slider(1, 3, value=2, step=0.5, label="Upscale (Mirage tiles and RapidOCR)")
            with gr.Accordion("Preprocessing", open=False):
                gray = gr.Checkbox(label="Grayscale")
                contrast = gr.Checkbox(value=True, label="Auto-contrast")
                invert = gr.Checkbox(label="Invert if image is dark")
                sharpen = gr.Checkbox(label="Sharpen")
            with gr.Accordion("Filters", open=False):
                min_obj = gr.Slider(0, 1, value=0.5, step=0.05, label="Min object score")
                min_mir = gr.Slider(0, 1, value=0.3, step=0.05, label="Min Mirage text score")
                min_rap = gr.Slider(0, 1, value=0.5, step=0.05, label="Min RapidOCR score")
                mode = gr.Radio(["0-1000 normalized", "pixels"], value="0-1000 normalized",
                                label="Box coordinates (switch if boxes look misplaced)")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            status = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Text"):
                    txt = textbox("Recognized text", 18)
                with gr.Tab("Annotated"):
                    ann = gr.Image(label="Red = text, green = objects", type="pil")
                with gr.Tab("Summary"):
                    summ = textbox("Scene and objects", 12)
                with gr.Tab("JSON"):
                    js = gr.JSON()
                    dl = gr.File(label="Download JSON")
    btn.click(run,
              [inp, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap],
              [ann, txt, summ, js, dl, status])


def main():
    ap = argparse.ArgumentParser(description="Mirage GUI")
    ap.add_argument("--port", type=int, default=None,
                    help="port to use (default: first free port starting at 7860)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
