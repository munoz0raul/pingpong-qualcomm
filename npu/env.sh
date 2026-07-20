#!/bin/bash
# npu/env.sh — the SINGLE place to configure your machine for Steps 7-8 (the NPU part).
#
# This file is SOURCED by the other npu/*.sh scripts (not run directly). Edit the two
# paths below to match your x86 Linux box, then run the scripts as-is — no other edits.
#
# Why a config file: the QAIRT SDK is a separate (free) download and lives somewhere
# different on every machine, so its path can't be baked into each script. Set it once here.

# --- EDIT THESE TWO ---------------------------------------------------------
# Your QAIRT SDK install (contains bin/qairt-converter, qairt-quantizer, etc.):
: "${SDK:=/path/to/qairt/2.47.0.260601}"
# aarch64 cross-compile sysroot — ONLY needed by build_base.sh / build_daemon.sh
# (where aarch64-linux-gnu-g++-13 and its libs live). Leave as-is if you're not
# rebuilding the C++ daemon.
: "${R:=/path/to/cross/root}"
# ----------------------------------------------------------------------------

# x86 work directory: where best.onnx, the DLCs, calib/, and the ctx output live.
# Defaults to the directory you run the script from — put best.onnx there and you're set.
: "${WORK:=$PWD}"

# Sanity: the SDK path must exist, or every downstream command fails cryptically.
if [ ! -d "$SDK" ]; then
  echo "ERROR: QAIRT SDK not found at: $SDK" >&2
  echo "       Edit npu/env.sh and set SDK=<your QAIRT install>." >&2
  return 1 2>/dev/null || exit 1
fi

# Put the SDK's tools on PATH. Its env script sets PATH/LD_LIBRARY_PATH for qairt-*.
# Newer SDKs ship bin/envsetup.sh; point QAIRT_ENV at yours if it's named differently.
: "${QAIRT_ENV:=$SDK/bin/envsetup.sh}"
if [ -f "$QAIRT_ENV" ]; then
  # shellcheck disable=SC1090
  source "$QAIRT_ENV"
fi

# Warn (don't fail) if the tools still aren't callable — the reader may source their own env.
if ! command -v qairt-converter >/dev/null 2>&1; then
  echo "warn: qairt tools not on PATH — source your SDK env, or set QAIRT_ENV in npu/env.sh" >&2
fi
