# Mirage GUI

A small local web GUI for the [Cactus Mirage](https://huggingface.co/Cactus-Compute/mirage)
vision model, with an optional [RapidOCR](https://github.com/RapidAI/RapidOCR) engine for
better text recognition.

Unofficial project, not affiliated with Cactus Compute. Everything runs on your machine.

## Features

- Upload an image or paste one from the clipboard
- Scene description, object detection and text recognition from Mirage
- Optional RapidOCR as a second text engine ("Both" merges the two)
- Tiling for Mirage: the image is cut into overlapping tiles and each tile is zoomed, which helps with small text (RapidOCR always reads the whole image)
- Preprocessing: auto-contrast, grayscale, invert-if-dark, sharpen
- Annotated image with boxes, text grouped into reading-order lines, copy button, JSON download
- One-command installer, app-menu shortcut, optional desktop icon

## Install (Linux)

```bash
git clone https://github.com/ashardian/mirage-gui.git
cd mirage-gui
./install.sh
```

The installer:

1. creates a virtual environment in `.venv`
2. installs Mirage (`cactus-compute`), Gradio, Pillow and NumPy
3. installs RapidOCR (falls back to the newer `rapidocr` package if the first one fails)
4. downloads the Mirage model so the first launch is fast
5. creates an app-menu entry called **Mirage GUI**

Options:

```bash
./install.sh --desktop-icon   # also put a shortcut on your Desktop
./install.sh --no-rapidocr    # skip RapidOCR (Mirage-only mode)
./install.sh --no-download    # do not pre-download the model
```

On Debian, Ubuntu and Linux Mint you may first need:

```bash
sudo apt install python3 python3-venv python3-pip
```

## Run

- Click **Mirage GUI** in your app menu (or the desktop icon), or
- run `./run.sh` from the project folder. No need to activate the virtual environment.

The page opens in your browser at `http://127.0.0.1:7860` (the next free port is used if
7860 is busy). It is only reachable from your own computer.

Options: `./run.sh --port 7861` and `./run.sh --no-browser`.

Stop it with `Ctrl+C` in the terminal window, or by closing that window.

## Desktop shortcut

`./install.sh` creates the shortcut for you. To make one by hand, save this as
`~/.local/share/applications/mirage-gui.desktop` (change the two paths to your clone):

```ini
[Desktop Entry]
Type=Application
Name=Mirage GUI
Exec="/home/YOU/mirage-gui/run.sh"
Icon=/home/YOU/mirage-gui/assets/mirage-gui.svg
Terminal=true
Categories=Graphics;Utility;
```

Then run `chmod +x ~/.local/share/applications/mirage-gui.desktop`. For a desktop icon, copy
the file to `~/Desktop`, make it executable, and if your desktop asks, choose
"Allow launching" from the right-click menu.

## Manual install (without the script)

```bash
git clone https://github.com/ashardian/mirage-gui.git
cd mirage-gui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rapidocr.txt     # optional, better text accuracy
python mirage_gui.py
```

If `rapidocr-onnxruntime` will not install on your Python version, use the newer package:

```bash
pip install rapidocr onnxruntime
```

## Using it

1. Add an image (upload, drag in, or paste).
2. Leave **Text engine** on "Both" if RapidOCR is ready, otherwise "Mirage".
3. Leave **Mirage tiles** on 2x2. Try 3x3 for very small text or "1x1" to compare. Tiles only affect Mirage.
4. Click **Run**, then check the **Text**, **Annotated**, **Summary** and **JSON** tabs.

Tips:

- Mirage is a tiny model and its own text reading is weak on small or low-contrast text.
  RapidOCR is usually the better text engine; Mirage is better for scene and object info.
- Under **Filters**, "Box coordinates" is set to 0-1000 normalized, which matches the
  output seen so far. If the boxes on the **Annotated** tab are in the wrong place, switch to "pixels".
- With Mirage tiles, text on a tile edge can show up as a fragment. The overlap and de-duplication
  catch most of these but not all. RapidOCR is not affected because it reads the whole image.

## Troubleshooting

**The page says "RapidOCR: not available".** The reason is printed in the terminal at startup
and shown on the page. Check it with the project's own Python:

```bash
.venv/bin/python -m pip list | grep -iE "rapidocr|onnx"
.venv/bin/python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('ok')"
```

If the package is missing, install it into this environment:
`.venv/bin/python -m pip install rapidocr-onnxruntime` (or `rapidocr onnxruntime`).
A very common cause is installing RapidOCR into a different Python than the one the app uses.

**`ensurepip` / `venv` errors during install.** Run `sudo apt install python3-venv python3-pip`.

**Model download fails.** Check your connection and run `./install.sh` again, or just launch
the app; it retries on first use.

**Changing the port.** `./run.sh --port 7861`.

## Update and uninstall

```bash
git pull && ./install.sh      # update
./uninstall.sh                # remove shortcuts and, if you agree, the virtual environment
```

## Privacy

Images are processed on your machine. The app binds to `127.0.0.1` only. Do not change the
launch settings to expose it on your network unless you know what you are doing. The Mirage model
and RapidOCR models are downloaded once from their publishers.

## Credits and license

- Mirage model by Cactus Compute (Apache-2.0 according to its model card)
- [Gradio](https://www.gradio.app/), [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/),
  [RapidOCR](https://github.com/RapidAI/RapidOCR)

This project is released under the MIT License (see `LICENSE`).
