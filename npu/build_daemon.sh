#!/bin/bash
# Recompile the SampleApp WITH runDaemon() + --daemon flags (Phase D).
# Same cross-aarch64 recipe validated in build_base.sh, in-place under daemon/.
set +e
SDK=/local/mnt/workspace/qairt/qairt/2.47.0.260601
R=/local/mnt/workspace/qairt/cross/root
export LD_LIBRARY_PATH="$R/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
GXX="$R/usr/bin/aarch64-linux-gnu-g++-13"
GCCDIR="$R/usr/lib/gcc-cross/aarch64-linux-gnu/13"

WORK=/local/mnt/workspace/qairt/daemon
cd "$WORK" || exit 1

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
  -o "$WORK/qnn-daemon-aarch64" \
  -ldl -static-libstdc++ -static-libgcc 2>&1 | head -50
echo "rc=${PIPESTATUS[0]}"
file "$WORK/qnn-daemon-aarch64" 2>&1
ls -la "$WORK/qnn-daemon-aarch64" 2>&1
echo "===== END ====="
