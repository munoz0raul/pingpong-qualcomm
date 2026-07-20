#!/bin/bash
# npu/env.sh — the SINGLE place to configure your machine for Steps 7-8 (the NPU part).
#
# This file is SOURCED by the other npu/*.sh scripts (not run directly). Edit the paths
# below to match your x86 Linux box, then run the scripts as-is — no other edits.
#
# Why a config file: the QAIRT SDK is a separate (free) download and lives somewhere
# different on every machine, so its path can't be baked into each script. Set it once here.

# --- EDIT THESE -------------------------------------------------------------
# QW = optional shortcut. If you followed Step 7.0 you already did `export QW=<work dir>`;
#   since it's exported, this file (sourced by the scripts) sees it too. When QW is set, the
#   four paths below AUTO-DERIVE from it and you edit NOTHING here. If QW is unset (e.g. a
#   fresh shell, or a hand-built layout), fall through to the manual defaults and edit those.
if [ -n "$QW" ]; then
  : "${SDK:=$QW/qairt/2.47.0.260601}"          # 7.0 step 1 unzips the SDK here
  : "${VENV:=$QW/.venv}"                        # 7.0 step 2 makes the venv here
  : "${LLVM_LIBS:=$QW/llvm-libs/usr/lib/llvm-18/lib}"   # 7.0 step 4 stages libc++ here
  : "${WORK:=$QW}"                              # best.onnx + calib/ get rsync'd here
fi

# SDK = the folder where you unzipped the QAIRT SDK. It holds the NPU compiler tools
#   (bin/.../qairt-converter, qairt-quantizer, qnn-context-binary-generator) and the
#   runtime libraries. Don't have it yet? See Step 7.0 in docs/REPRODUCE.md — it's a
#   free download. After unzipping, this is the versioned folder inside, e.g.
#   /home/you/qairt/2.47.0.260601  (the one that contains bin/, lib/, include/).
: "${SDK:=/path/to/qairt/2.47.0.260601}"

# R = the ROOT of your aarch64 cross-compiler (a "sysroot"). Only needed if you rebuild
#   the live C++ daemon (build_base.sh / build_daemon.sh) — the compiler that makes ARM
#   binaries for the board lives here (usr/bin/aarch64-linux-gnu-g++-13). If you only
#   want the one-shot NPU test (Step 8.1), leave this untouched — it's not used.
: "${R:=/path/to/cross/root}"

# VENV = a Python virtualenv the SDK tools run inside (they need numpy==1.26.4 etc.).
#   Leave empty to use system python. On Ubuntu the stock `python3 -m venv` may be broken
#   (no ensurepip) — Step 7.0 shows the `virtualenv` fallback. Point this at the venv's
#   root (the dir that contains bin/activate).
: "${VENV:=}"

# LLVM_LIBS = a dir holding libc++.so.1, libc++abi.so.1 and libunwind.so.1. The SDK's
#   native .so files are built against LLVM's libc++, which a clean Ubuntu box does NOT
#   ship — without these you get "libc++.so.1: cannot open shared object file". Step 7.0
#   shows how to fetch them with apt-get download + dpkg-deb -x (no sudo). Leave empty if
#   your distro already provides libc++ system-wide.
: "${LLVM_LIBS:=}"
# ----------------------------------------------------------------------------

# WORK = your working directory on this x86 box: where you put best.onnx and the calib/
#   folder, and where the DLCs + context .bin get written. Defaults to the folder you run
#   the script from, so the simple recipe is: cd into a folder, drop best.onnx there, go.
: "${WORK:=$PWD}"

# Sanity: the SDK path must exist, or every downstream command fails cryptically.
if [ ! -d "$SDK" ]; then
  echo "ERROR: QAIRT SDK not found at: $SDK" >&2
  echo "       Edit npu/env.sh and set SDK=<your QAIRT install>." >&2
  return 1 2>/dev/null || exit 1
fi

# Activate the venv first (so its python + numpy are what the SDK tools import).
if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090,SC1091
  source "$VENV/bin/activate"
fi

# Put the SDK's tools on PATH. Its env script sets PATH/LD_LIBRARY_PATH for qairt-*.
# Newer SDKs ship bin/envsetup.sh; point QAIRT_ENV at yours if it's named differently.
: "${QAIRT_ENV:=$SDK/bin/envsetup.sh}"
if [ -f "$QAIRT_ENV" ]; then
  # shellcheck disable=SC1090
  source "$QAIRT_ENV"
fi

# Prepend the LLVM runtime libs the SDK's native .so files need (clean Ubuntu lacks them).
if [ -n "$LLVM_LIBS" ] && [ -d "$LLVM_LIBS" ]; then
  export LD_LIBRARY_PATH="$LLVM_LIBS:$LD_LIBRARY_PATH"
fi

# Warn (don't fail) if the tools still aren't callable — the reader may source their own env.
if ! command -v qairt-converter >/dev/null 2>&1; then
  echo "warn: qairt tools not on PATH — source your SDK env, or set QAIRT_ENV in npu/env.sh" >&2
fi
