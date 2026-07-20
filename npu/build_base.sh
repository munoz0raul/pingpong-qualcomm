#!/bin/bash
# Cross-compile the BASE QNN SampleApp (unmodified) with our aarch64 toolchain,
# just to prove that the QNN headers + toolchain produce a binary that runs on the board.
set +e
source "$(dirname "$0")/env.sh" || exit 1
# Needs the aarch64 cross-compiler ($R) — the one path env.sh doesn't derive from $QW.
if [ ! -x "$R/usr/bin/aarch64-linux-gnu-g++-13" ]; then
  echo "ERROR: aarch64 cross-compiler not found under R=$R" >&2
  echo "       Set R in npu/env.sh. See 'Set up the aarch64 cross-compiler' in Step 8.2." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$R/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
GXX="$R/usr/bin/aarch64-linux-gnu-g++-13"
GCCDIR="$R/usr/lib/gcc-cross/aarch64-linux-gnu/13"

BUILD="$WORK/daemon"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp -r "$SDK/examples/QNN/SampleApp/SampleApp/"* "$BUILD/" 2>/dev/null
cd "$BUILD" || exit 1
echo "copied to $BUILD; sources:"; find src -name '*.cpp' | wc -l

# sources their Makefile links (main + QnnSampleApp + Log + PAL linux/common + Utils + WrapperUtils)
SRCS="src/main.cpp src/QnnSampleApp.cpp \
$(ls src/Log/*.cpp 2>/dev/null) \
$(ls src/PAL/src/linux/*.cpp 2>/dev/null) \
$(ls src/PAL/src/common/*.cpp 2>/dev/null) \
$(ls src/Utils/*.cpp 2>/dev/null) \
$(ls src/WrapperUtils/*.cpp 2>/dev/null)"

INCLUDES="-Isrc -Isrc/Log -Isrc/Utils -Isrc/WrapperUtils -Isrc/PAL/include -I$SDK/include/QNN"

echo "===== compiling (c++17, no -Werror/-pg/-flto) ====="
"$GXX" -std=c++17 --sysroot="$R" -B "$GCCDIR" \
  -L "$R/usr/aarch64-linux-gnu/lib" -L "$GCCDIR" \
  -fPIC -Wno-write-strings -fno-exceptions -fno-rtti -DQNN_API= \
  $INCLUDES $SRCS \
  -o "$BUILD/qnn-sample-app-aarch64" \
  -ldl -static-libstdc++ -static-libgcc 2>&1 | head -40
echo "rc=${PIPESTATUS[0]}"
file "$BUILD/qnn-sample-app-aarch64" 2>&1
ls -la "$BUILD/qnn-sample-app-aarch64" 2>&1
echo "===== END ====="
