#!/usr/bin/env python3
"""Mirage GUI - a small local web UI for the Cactus Mirage vision model.

Unofficial project, not affiliated with Cactus Compute.
Everything runs locally; the page is only reachable from this machine.
"""
import argparse
import html
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np
import cactus
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter, ImageOps

__version__ = "1.2.0"

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
print("RapidOCR ready" if RAPID else f"RapidOCR unavailable: {RAPID_ERR}")


# ---------- image helpers ----------
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


# ---------- engines ----------
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


# ---------- merging and output helpers ----------
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


# ---------- core analysis ----------
def analyze(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    """Run the selected engines on one PIL image. Returns a dict with texts, objects and scene."""
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

    # Mirage tiles zoom into small text
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
    return {"orig": orig, "texts": texts, "objs": objs, "scene": scene, "note": note,
            "tiles": n * n if (want_mirage and n > 1) else 1}


def result_data(a):
    return {
        "scene": a["scene"],
        "objects": [{"name": o["name"], "score": round(o["score"], 3),
                     "bbox": [round(x) for x in o["bbox"]], "description": o["desc"]} for o in a["objs"]],
        "text": [{"string": t["string"], "score": round(t["score"], 3),
                  "bbox": [round(x) for x in t["bbox"]]} for t in a["texts"]],
    }


def write_json(data, directory=None):
    fd, path = tempfile.mkstemp(suffix=".json", dir=directory)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


# ---------- clipboard and screen capture (optional system tools) ----------
def copy_to_clipboard(text):
    """Copy text with wl-copy, xclip or xsel if one is installed. Returns the tool name or None."""
    for argv in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
        if shutil.which(argv[0]):
            try:
                subprocess.run(argv, input=text.encode("utf-8"), timeout=5, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return argv[0]
            except Exception:  # noqa: BLE001
                continue
    return None


def detect_capture_tools():
    """Find installed region-screenshot tools. Returns {name: builder(path) -> (argv, output_on_stdout)}."""
    have = shutil.which
    wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    tools = {}
    if have("maim"):
        tools["maim"] = lambda p: (["maim", "-s", p], False)
    if have("flameshot"):
        tools["flameshot"] = lambda p: (["flameshot", "gui", "--raw"], True)
    if have("gnome-screenshot"):
        tools["gnome-screenshot"] = lambda p: (["gnome-screenshot", "-a", "-f", p], False)
    if have("spectacle"):
        tools["spectacle"] = lambda p: (["spectacle", "-r", "-b", "-n", "-o", p], False)
    if have("scrot"):
        tools["scrot"] = lambda p: (["scrot", "-s", p], False)
    if wayland and have("grim") and have("slurp"):
        tools["grim+slurp"] = lambda p: (["sh", "-c", 'grim -g "$(slurp)" "$1"', "sh", p], False)
    order = (["grim+slurp", "spectacle", "gnome-screenshot", "flameshot"] if wayland
             else ["maim", "flameshot", "gnome-screenshot", "spectacle", "scrot"])
    ordered = {k: tools[k] for k in order if k in tools}
    ordered.update({k: v for k, v in tools.items() if k not in ordered})
    return ordered


def capture(delay, tool_name):
    """Let the user pick a screen region with a system tool. Returns (PIL image or None, message)."""
    tools = detect_capture_tools()
    if not tools:
        return None, ("No screenshot tool found. Install one: X11 `sudo apt install maim`, "
                      "Wayland `sudo apt install grim slurp`, then restart the app.")
    name = tool_name if tool_name in tools else next(iter(tools))
    time.sleep(max(0, float(delay)))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "capture.png")
    try:
        argv, to_stdout = tools[name](path)
        try:
            if to_stdout:
                with open(path, "wb") as f:
                    r = subprocess.run(argv, stdout=f, stderr=subprocess.PIPE, timeout=180)
            else:
                r = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
        except Exception as e:  # noqa: BLE001
            return None, f"Capture with {name} failed: {e}"
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            err = (r.stderr or b"").decode("utf-8", "ignore").strip()[:200]
            return None, f"Capture cancelled or failed ({name}). {err}".strip()
        img = Image.open(path)
        img.load()
        return img.convert("RGB"), f"Captured with {name}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- UI callbacks ----------
def run(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    if img is None:
        return None, "", "", None, None, "Upload, paste or capture an image first."
    t0 = time.time()
    a = analyze(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap)
    summary = []
    if a["scene"]:
        summary.append("SCENE: " + ", ".join(f"{k}: {v}" for k, v in a["scene"].items()))
    for o in a["objs"]:
        desc = ", ".join(f"{k}={v}" for k, v in o["desc"].items())
        summary.append(f"- {o['name']} ({o['score']:.2f}) {desc}".rstrip())
    data = result_data(a)
    status = (f"{len(a['texts'])} text boxes, {len(a['objs'])} objects | Mirage tiles: {a['tiles']} | "
              f"coords: {mode} | {time.time() - t0:.1f}s{a['note']}")
    return (annotate(a["orig"], a["texts"], a["objs"]), to_lines(a["texts"]), "\n".join(summary),
            data, write_json(data), status)


def _with_clipboard(out, copy):
    """Append clipboard info to the status of a run() result and copy the text if asked."""
    if not copy or not out[1]:
        return out
    tool = copy_to_clipboard(out[1])
    note = (f" | copied to clipboard ({tool})" if tool
            else " | clipboard tool not found (install xclip, xsel or wl-clipboard)")
    return out[:5] + (out[5] + note,)


def run_ui(img, copy, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    out = run(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap)
    return _with_clipboard(out, copy)


def capture_and_run(delay, tool, autorun, copy, mode, grid, up, engine, gray, contrast, invert, sharpen,
                    min_obj, min_mir, min_rap):
    img, msg = capture(delay, tool)
    if img is None:
        return None, None, "", "", None, None, msg
    if not autorun:
        return img, None, "", "", None, None, msg + ". Press Run to read it."
    out = run(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap)
    out = _with_clipboard(out, copy)
    return (img,) + out[:5] + (f"{msg} | {out[5]}",)


def batch(files, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    if not files:
        return "", None, "Add one or more images first."
    t0 = time.time()
    out_dir = tempfile.mkdtemp()
    parts, results = [], {}
    for f in files:
        path = getattr(f, "name", f)
        name = os.path.basename(str(path))
        try:
            a = analyze(Image.open(path), mode, grid, up, engine, gray, contrast, invert, sharpen,
                        min_obj, min_mir, min_rap)
            parts.append(f"=== {name} ===\n{to_lines(a['texts'])}\n")
            results[name] = result_data(a)
        except Exception as e:  # noqa: BLE001
            parts.append(f"=== {name} ===\n[error: {e}]\n")
            results[name] = {"error": str(e)}
    text = "\n".join(parts)
    txt_path = os.path.join(out_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    json_path = write_json(results, out_dir)
    return text, [txt_path, json_path], f"Processed {len(files)} image(s) in {time.time() - t0:.1f}s"


def compare(img, mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap):
    """Run Mirage, RapidOCR and Both on the same image for a side-by-side comparison."""
    if img is None:
        return "", "", "", "Add an image in the Image tab first."
    t0 = time.time()
    outs = []
    for eng in ("Mirage", "RapidOCR", "Both"):
        if eng != "Mirage" and RAPID is None:
            outs.append("(RapidOCR is not available)")
            continue
        a = analyze(img, mode, grid, up, eng, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap)
        outs.append(to_lines(a["texts"]) or "(no text found)")
    return outs[0], outs[1], outs[2], f"Compared in {time.time() - t0:.1f}s"


# ---------- UI ----------
def _ver(name):
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return "not installed"


def about_text():
    methods = [m for m in dir(model) if not m.startswith("_") and callable(getattr(model, m, None))]
    clip = next((c for c in ("wl-copy", "xclip", "xsel") if shutil.which(c)), "none found")
    rows = [
        ("Mirage GUI", __version__),
        ("Python", platform.python_version()),
        ("Gradio", gr.__version__),
        ("cactus-compute", _ver("cactus-compute")),
        ("RapidOCR", ("rapidocr-onnxruntime " + _ver("rapidocr-onnxruntime")
                      if _ver("rapidocr-onnxruntime") != "not installed" else "rapidocr " + _ver("rapidocr"))
         if RAPID else f"not available ({RAPID_ERR or 'not installed'})"),
        ("onnxruntime", _ver("onnxruntime")),
        ("Session type", os.environ.get("XDG_SESSION_TYPE", "unknown")),
        ("Screenshot tools", ", ".join(detect_capture_tools()) or "none found"),
        ("Clipboard tool", clip),
        ("Mirage methods", ", ".join(methods) or "unknown"),
    ]
    return "| Item | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def textbox(label, lines):
    try:
        return gr.Textbox(label=label, lines=lines, max_lines=lines, show_copy_button=True)
    except TypeError:  # newer Gradio versions renamed this option
        return gr.Textbox(label=label, lines=lines, max_lines=lines)


_BADGE = ("display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8em;"
          "background:{bg};color:{fg}")
if RAPID:
    HEADER_HTML = (
        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        '<h2 style="margin:0">Mirage GUI</h2>'
        f'<span style="{_BADGE.format(bg="#16a34a22", fg="#16a34a")}">RapidOCR ready</span></div>'
    )
else:
    HEADER_HTML = (
        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        '<h2 style="margin:0">Mirage GUI</h2>'
        f'<span style="{_BADGE.format(bg="#d9770622", fg="#d97706")}">RapidOCR not available (Mirage only)</span></div>'
        f'<div style="font-size:0.8em;opacity:0.75;margin-top:4px">Reason: {html.escape(RAPID_ERR or "not installed")}</div>'
    )

_A = 'target="_blank" rel="noopener"'
_CREDITS = [
    ("Cactus Compute", "https://cactuscompute.com"),
    ("Mirage model", "https://huggingface.co/Cactus-Compute/mirage"),
    ("cactus-compute runtime", "https://pypi.org/project/cactus-compute/"),
    ("RapidOCR", "https://github.com/RapidAI/RapidOCR"),
    ("PaddleOCR models", "https://github.com/PaddlePaddle/PaddleOCR"),
    ("ONNX Runtime", "https://onnxruntime.ai"),
    ("Gradio", "https://www.gradio.app"),
    ("Hugging Face", "https://huggingface.co"),
    ("Pillow", "https://python-pillow.org"),
    ("NumPy", "https://numpy.org"),
    ("OpenCV", "https://opencv.org"),
]
_credit_links = "".join(f'<a href="{u}" {_A}>{n}</a>' for n, u in _CREDITS)
FOOTER_HTML = (
    '<div style="text-align:center;font-size:0.85em;opacity:0.85;padding:14px 0 4px;line-height:1.6">'
    f'<div style="font-size:1.1em;margin-bottom:8px">Made with ❤️ by <a href="https://github.com/ashardian" {_A}><b>Ashar Dian</b></a></div>'
    '<div style="opacity:0.7;margin-bottom:2px">Credits</div>'
    f'<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:2px 18px">{_credit_links}</div>'
    '<div style="margin-top:8px;opacity:0.8">Unofficial project, not affiliated with any of the above. '
    'Everything runs locally on your machine.</div></div>'
)

_tools = list(detect_capture_tools())

with gr.Blocks(title="Mirage GUI") as demo:
    gr.HTML(HEADER_HTML)

    # ===== Advanced options: hidden until the user opens the panel =====
    with gr.Accordion("⚙ Advanced options (click to show or hide)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Text recognition**")
                engine = gr.Radio(["Mirage", "RapidOCR", "Both"], value="Both" if RAPID else "Mirage",
                                  label="Text engine",
                                  info="Both = RapidOCR plus Mirage, usually the most accurate.")
                grid = gr.Radio(["1x1 (whole)", "2x2", "3x3"], value="2x2", label="Mirage tiles",
                                info="Zooms into pieces of the image for small text. Mirage only.")
                up = gr.Slider(1, 3, value=2, step=0.5, label="Upscale",
                               info="Bigger helps small text but is slower.")
            with gr.Column():
                gr.Markdown("**Image cleanup**")
                contrast = gr.Checkbox(value=True, label="Auto-contrast")
                gray = gr.Checkbox(label="Grayscale")
                invert = gr.Checkbox(label="Invert dark images",
                                     info="Helps light text on a dark background.")
                sharpen = gr.Checkbox(label="Sharpen")
            with gr.Column():
                gr.Markdown("**Filters**")
                min_obj = gr.Slider(0, 1, value=0.5, step=0.05, label="Object confidence",
                                    info="Hide objects scoring below this.")
                min_mir = gr.Slider(0, 1, value=0.3, step=0.05, label="Mirage text confidence")
                min_rap = gr.Slider(0, 1, value=0.5, step=0.05, label="RapidOCR text confidence")
                mode = gr.Radio(["0-1000 normalized", "pixels"], value="0-1000 normalized",
                                label="Box coordinates",
                                info="Switch to pixels if boxes look misplaced.")
        gr.Markdown("**Screen capture**")
        with gr.Row():
            tool = gr.Dropdown(choices=_tools or ["none found"], value=_tools[0] if _tools else "none found",
                               label="Capture tool")
            delay = gr.Slider(0, 10, value=3, step=1, label="Delay before capture (seconds)",
                              info="Time to switch to the window you want.")
            autorun = gr.Checkbox(value=True, label="Read the text right after capture")
    settings = [mode, grid, up, engine, gray, contrast, invert, sharpen, min_obj, min_mir, min_rap]

    # ===== Main screen =====
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    inp = gr.Image(type="pil", label="Image", sources=["upload", "clipboard"], height=330)
                    with gr.Row():
                        btn = gr.Button("Run", variant="primary", scale=2)
                        cap_btn = gr.Button("📸 Capture region", scale=1)
                    copy_cb = gr.Checkbox(label="Copy the text to the clipboard after reading")
                with gr.Column(scale=6):
                    status = gr.Markdown("Add an image and click **Run**, or click **Capture region**.")
                    with gr.Tabs():
                        with gr.Tab("Text"):
                            txt = textbox("Recognized text", 14)
                        with gr.Tab("Annotated"):
                            ann = gr.Image(label="Red = text, green = objects", type="pil", height=330)
                        with gr.Tab("Summary"):
                            summ = textbox("Scene and objects", 12)
                        with gr.Tab("JSON"):
                            js = gr.JSON()
                            dl = gr.File(label="Download JSON")

        with gr.Tab("Batch"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gr.Markdown("Read many images at once. Uses the settings in **Advanced options**.")
                    files = gr.File(file_count="multiple", file_types=["image"], label="Images")
                    bbtn = gr.Button("Process all", variant="primary")
                    bstatus = gr.Markdown()
                with gr.Column(scale=6):
                    btxt = textbox("Combined text", 14)
                    bdl = gr.File(file_count="multiple", label="Download results (txt and json)")

        with gr.Tab("Compare engines"):
            gr.Markdown("Runs the image from the **Image** tab through each text engine so you can see "
                        "which one reads it best.")
            cbtn = gr.Button("Compare", variant="primary")
            cstatus = gr.Markdown()
            with gr.Row():
                c1 = textbox("Mirage only", 14)
                c2 = textbox("RapidOCR only", 14)
                c3 = textbox("Both (merged)", 14)

        with gr.Tab("About"):
            gr.Markdown(about_text())

    outs = [ann, txt, summ, js, dl, status]
    btn.click(run_ui, [inp, copy_cb] + settings, outs, api_name="run")
    cap_btn.click(capture_and_run, [delay, tool, autorun, copy_cb] + settings, [inp] + outs,
                  api_name="capture")
    bbtn.click(batch, [files] + settings, [btxt, bdl, bstatus], api_name="batch")
    cbtn.click(compare, [inp] + settings, [c1, c2, c3, cstatus], api_name="compare")

    gr.HTML(FOOTER_HTML)


def main():
    ap = argparse.ArgumentParser(description="Mirage GUI")
    ap.add_argument("--port", type=int, default=None,
                    help="port to use (default: first free port starting at 7860)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
