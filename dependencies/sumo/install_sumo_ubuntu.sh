#!/bin/bash
# SUMO sanity / install helper for user-mode (no sudo) environments.
#
# The actual SUMO traffic simulator is now installed as a pip wheel via
# `eclipse-sumo` in pixi.toml. That means:
#
#   pixi install
#
# is the one-shot install. This script is a complementary sanity-check
# that confirms the wheel landed, finds the bundled SUMO_HOME path, and
# offers to append the right export to ~/.bashrc for non-pixi shells.
#
# No sudo, no apt, no PPA. Works on Ubuntu/Debian, RHEL, Arch, anywhere
# pixi runs.
#
# Usage:
#   bash dependencies/sumo/install_sumo_ubuntu.sh
#
# Flags:
#   --skip-shellrc     Don't offer to append SUMO_HOME to ~/.bashrc.
#   --print-only       Print the SUMO_HOME path and exit (for scripting).

set -euo pipefail

WRITE_SHELLRC=1
PRINT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-shellrc) WRITE_SHELLRC=0 ;;
    --print-only)   PRINT_ONLY=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# //; s/^#//' | head -30
      exit 0
      ;;
    *)
      echo "[sumo-sanity] unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# 1. Make sure pixi is available.
if ! command -v pixi >/dev/null 2>&1; then
  cat >&2 <<EOF
[sumo-sanity] pixi not found on PATH.
  This project uses pixi to manage the SUMO install. Install pixi first:
    https://pixi.sh/latest/#installation
  then run \`pixi install\` from the repo root before re-running this script.
EOF
  exit 1
fi

# 2. Make sure eclipse-sumo is in the pixi env.
SUMO_PATH="$(pixi run python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
if [ -z "$SUMO_PATH" ] || [ ! -d "$SUMO_PATH" ]; then
  cat >&2 <<EOF
[sumo-sanity] eclipse-sumo not installed in pixi env yet.
  Run \`pixi install\` from the repo root to pull the wheel (~150 MB Linux x86_64).
  Then re-run this script.
EOF
  exit 1
fi

# 3. Locate the bundled SUMO binary via sumolib (most reliable).
SUMO_BIN="$(pixi run python -c 'import sumolib; print(sumolib.checkBinary("sumo"))' 2>/dev/null || true)"

if [ "$PRINT_ONLY" = 1 ]; then
  echo "SUMO_HOME=$SUMO_PATH"
  echo "SUMO_BIN=$SUMO_BIN"
  exit 0
fi

echo "[sumo-sanity] eclipse-sumo found:"
echo "  SUMO_HOME = $SUMO_PATH"
echo "  binary    = $SUMO_BIN"

# 4. Sanity: invoke the binary.
echo
echo "[sumo-sanity] sumo --version:"
pixi run sumo --version 2>&1 | sed 's/^/  /' | head -5 || {
  echo "[sumo-sanity] WARN: pixi run sumo --version failed; check PATH inside the pixi env"
}

# 5. Verify Python imports.
echo
echo "[sumo-sanity] python import check:"
pixi run python -c "
import sumo, sumolib, traci
print(f'  sumo package version (best-effort):', getattr(sumo, '__version__', 'n/a'))
print(f'  sumolib version: {sumolib.__version__}')
print(f'  traci version: {traci.__version__}')
" 2>&1 | sed 's/^/  /'

# 6. Optionally update ~/.bashrc for non-pixi shells.
if [ "$WRITE_SHELLRC" = 1 ]; then
  BASHRC="$HOME/.bashrc"
  MARKER="# OpenCDA-MARL: SUMO env (added by dependencies/sumo/install_sumo_ubuntu.sh)"
  if [ -f "$BASHRC" ] && ! grep -qF "$MARKER" "$BASHRC"; then
    {
      echo ""
      echo "$MARKER"
      echo "# Inside the pixi env the engine finds SUMO through the eclipse-sumo package;"
      echo "# this block exports it in plain bash too, for tools outside the pixi env."
      echo "export SUMO_HOME=\"$SUMO_PATH\""
      echo "export PATH=\"\$SUMO_HOME/bin:\$PATH\""
    } >> "$BASHRC"
    echo
    echo "[sumo-sanity] appended SUMO_HOME export to $BASHRC"
    echo "  Source it with: source ~/.bashrc"
  fi
fi

echo
echo "[sumo-sanity] done. SUMO is ready for:"
echo "    pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl"
