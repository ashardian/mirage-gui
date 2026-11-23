#!/usr/bin/env bash
# Removes the Mirage GUI shortcuts and, if you agree, the virtual environment.
# The project folder itself and the downloaded Mirage model cache are left alone.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESK_DIR="$(xdg-user-dir DESKTOP 2> /dev/null || echo "$HOME/Desktop")"

rm -f "$APPS/mirage-gui.desktop" "$DESK_DIR/mirage-gui.desktop"
echo "Shortcuts removed."

if [ -d .venv ]; then
  read -rp "Delete the virtual environment (.venv)? [y/N] " a
  if [[ "$a" =~ ^[Yy]$ ]]; then
    rm -rf .venv
    echo ".venv deleted."
  fi
fi
echo "Done. You can delete this folder whenever you like."
