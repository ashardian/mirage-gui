#!/usr/bin/env bash
# Starts Mirage GUI using this folder's own virtual environment (no activation needed).
# Extra options are passed through, e.g.: ./run.sh --port 7861 --no-browser
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run ./install.sh first."
  read -rp "Press Enter to close."
  exit 1
fi

.venv/bin/python mirage_gui.py "$@"
code=$?

# 130 = Ctrl+C, 143 = terminated: normal ways to stop. Anything else is worth reading.
if [ "$code" -ne 0 ] && [ "$code" -ne 130 ] && [ "$code" -ne 143 ]; then
  echo
  echo "Mirage GUI exited with an error (code $code)."
  read -rp "Press Enter to close."
fi
exit "$code"
