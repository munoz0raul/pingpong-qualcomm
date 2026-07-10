#!/bin/bash
# Cross-compile the BASE QNN SampleApp (unmodified) with our aarch64 toolchain,
# just to prove that the QNN headers + toolchain produce a binary that runs on the board.
set +e
SDK=/local/mnt/workspace/qairt/qairt/2.47.0.260601
R=/local/mnt/workspace/qairt/cross/root
export LD_LIBRARY_PATH="$R/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
GXX="$R/usr/bin/aarch64-linux-gnu-g++-13"
GCCDIR="$R/usr/lib/gcc-cross/aarch64-linux-gnu/13"

WORK=/local/mnt/workspace/qairt/daemon
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r "$SDK/examples/QNN/SampleApp/SampleApp/"* "$WORK/" 2>/dev/null
cd "$WORK" || exit 1
echo "copied to $WORK; sources:"; find src -name '*.cpp' | wc -l

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
  -o "$WORK/qnn-sample-app-aarch64" \
  -ldl -static-libstdc++ -static-libgcc 2>&1 | head -40
echo "rc=${PIPESTATUS[0]}"
file "$WORK/qnn-sample-app-aarch64" 2>&1
ls -la "$WORK/qnn-sample-app-aarch64" 2>&1
echo "===== END ====="
