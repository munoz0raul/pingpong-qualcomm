#!/bin/bash
# Recompile the SampleApp WITH runDaemon() + --daemon flags (Phase D).
# Same cross-aarch64 recipe validated in build_base.sh, in-place under daemon/.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh" || exit 1
export LD_LIBRARY_PATH="$R/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
GXX="$R/usr/bin/aarch64-linux-gnu-g++-13"
GCCDIR="$R/usr/lib/gcc-cross/aarch64-linux-gnu/13"

BUILD="$WORK/daemon"

# --- one-time setup: lay down the SampleApp source tree, then overlay our modified files ---
# The repo ships only the 3 modified sources (npu/daemon/); the rest of the SampleApp tree
# (Log/ PAL/ Utils/ WrapperUtils/ + the stock .cpp) comes from the SDK. If src/ isn't there
# yet, copy it from the SDK — no manual step, no ordering trap.
if [ ! -d "$BUILD/src" ]; then
  SAMPLE="$SDK/examples/QNN/SampleApp/SampleApp"
  if [ ! -d "$SAMPLE" ]; then
    echo "ERROR: SampleApp not found at $SAMPLE" >&2
    echo "       Check your SDK layout, or copy the SampleApp tree to $BUILD/ by hand." >&2
    exit 1
  fi
  echo "===== first build: copying SampleApp from the SDK into $BUILD ====="
  mkdir -p "$BUILD"
  cp -r "$SAMPLE/"* "$BUILD/" || exit 1
fi
# Always overlay our modified files (keeps them in sync if the repo copy changed).
echo "===== overlaying modified daemon sources (runDaemon + --daemon flags) ====="
cp "$HERE/daemon/main.cpp"          "$BUILD/src/main.cpp"          || exit 1
cp "$HERE/daemon/QnnSampleApp.cpp"  "$BUILD/src/QnnSampleApp.cpp"  || exit 1
cp "$HERE/daemon/QnnSampleApp.hpp"  "$BUILD/src/QnnSampleApp.hpp"  || exit 1

cd "$BUILD" || exit 1

SRCS="src/main.cpp src/QnnSampleApp.cpp \
$(ls src/Log/*.cpp 2>/dev/null) \
$(ls src/PAL/src/linux/*.cpp 2>/dev/null) \
$(ls src/PAL/src/common/*.cpp 2>/dev/null) \
$(ls src/Utils/*.cpp 2>/dev/null) \
$(ls src/WrapperUtils/*.cpp 2>/dev/null)"

INCLUDES="-Isrc -Isrc/Log -Isrc/Utils -Isrc/WrapperUtils -Isrc/PAL/include -I$SDK/include/QNN"

echo "===== compiling daemon (c++17) ====="
"$GXX" -std=c++17 --sysroot="$R" -B "$GCCDIR" \
  -L "$R/usr/aarch64-linux-gnu/lib" -L "$GCCDIR" \
  -fPIC -Wno-write-strings -fno-exceptions -fno-rtti -DQNN_API= \
  $INCLUDES $SRCS \
  -o "$BUILD/qnn-daemon-aarch64" \
  -ldl -static-libstdc++ -static-libgcc 2>&1 | head -50
echo "rc=${PIPESTATUS[0]}"
file "$BUILD/qnn-daemon-aarch64" 2>&1
ls -la "$BUILD/qnn-daemon-aarch64" 2>&1
echo "===== END ====="
