#!/usr/bin/env bash
# Mirage GUI installer
#   - creates a Python virtual environment in ./.venv
#   - installs Mirage (cactus-compute), Gradio, Pillow, NumPy
#   - installs RapidOCR (better text engine) unless --no-rapidocr is given
#   - downloads the Mirage model now, so the first launch is fast
#   - creates an app-menu shortcut (and a desktop icon with --desktop-icon)
set -euo pipefail

usage() {
  cat << 'USAGE'
Usage: ./install.sh [options]

  --no-rapidocr    skip RapidOCR (Mirage-only mode)
  --no-download    do not pre-download the Mirage model
  --desktop-icon   also put a shortcut on your Desktop
  -h, --help       show this help
USAGE
}

WITH_RAPID=1
PREFETCH=1
DESKTOP_ICON=0
for arg in "$@"; do
  case "$arg" in
    --no-rapidocr) WITH_RAPID=0 ;;
    --no-download) PREFETCH=0 ;;
    --desktop-icon) DESKTOP_ICON=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$DIR"

echo "==> Checking Python"
if ! command -v python3 > /dev/null 2>&1; then
  echo "python3 not found. On Debian/Ubuntu/Mint: sudo apt install python3 python3-venv python3-pip"
  exit 1
fi
python3 --version
if ! python3 -c "import venv, ensurepip" > /dev/null 2>&1; then
  echo "The venv module is missing. On Debian/Ubuntu/Mint run:"
  echo "  sudo apt install python3-venv python3-pip"
  exit 1
fi

echo "==> Creating virtual environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip

echo "==> Installing Mirage and the GUI dependencies"
.venv/bin/python -m pip install -r requirements.txt

if [ "$WITH_RAPID" = 1 ]; then
  echo "==> Installing RapidOCR"
  if ! .venv/bin/python -m pip install -r requirements-rapidocr.txt; then
    echo "rapidocr-onnxruntime failed; trying the newer 'rapidocr' package"
    if ! .venv/bin/python -m pip install rapidocr onnxruntime; then
      echo "WARNING: could not install RapidOCR. The GUI still works in Mirage-only mode."
    fi
  fi
  if .venv/bin/python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" > /dev/null 2>&1 \
     || .venv/bin/python -c "from rapidocr import RapidOCR; RapidOCR()" > /dev/null 2>&1; then
    echo "RapidOCR check: OK"
  else
    echo "RapidOCR check: could not load it (the GUI will show the exact error at startup)"
  fi
fi

if [ "$PREFETCH" = 1 ]; then
  echo "==> Downloading the Mirage model (about 22 MB)"
  .venv/bin/python -c "import cactus; cactus.Mirage()" \
    || echo "WARNING: model download failed. It will be retried on the first launch."
fi

echo "==> Creating the app-menu shortcut"
chmod +x run.sh uninstall.sh
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/mirage-gui.desktop" << DESK
[Desktop Entry]
Type=Application
Version=1.0
Name=Mirage GUI
Comment=Local image understanding and OCR with Cactus Mirage
Exec="$DIR/run.sh"
Icon=$DIR/assets/mirage-gui.svg
Terminal=true
Categories=Graphics;Utility;
DESK
chmod +x "$APPS/mirage-gui.desktop"
if command -v update-desktop-database > /dev/null 2>&1; then
  update-desktop-database "$APPS" > /dev/null 2>&1 || true
fi

if [ "$DESKTOP_ICON" = 1 ]; then
  DESK_DIR="$(xdg-user-dir DESKTOP 2> /dev/null || echo "$HOME/Desktop")"
  mkdir -p "$DESK_DIR"
  cp "$APPS/mirage-gui.desktop" "$DESK_DIR/mirage-gui.desktop"
  chmod +x "$DESK_DIR/mirage-gui.desktop"
  if command -v gio > /dev/null 2>&1; then
    gio set "$DESK_DIR/mirage-gui.desktop" metadata::trusted true > /dev/null 2>&1 || true
  fi
  echo "Desktop icon created: $DESK_DIR/mirage-gui.desktop"
fi

echo "==> Optional tools for the Capture button and auto-copy"
have_any() { for c in "$@"; do command -v "$c" > /dev/null 2>&1 && return 0; done; return 1; }
if have_any maim flameshot gnome-screenshot spectacle scrot grim; then
  echo "Screenshot tool: found"
else
  echo "No screenshot tool found. For the Capture button install one:"
  echo "  X11:     sudo apt install maim"
  echo "  Wayland: sudo apt install grim slurp"
fi
if have_any xclip xsel wl-copy; then
  echo "Clipboard tool: found"
else
  echo "No clipboard tool found (only needed for auto-copy): sudo apt install xclip   (Wayland: wl-clipboard)"
fi

echo
echo "Done. Start it from your app menu (search for 'Mirage GUI') or run: $DIR/run.sh"
