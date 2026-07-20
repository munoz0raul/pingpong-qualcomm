#!/bin/bash
# npu/copy_runtime_libs.sh — stage the Qualcomm runtime .so libraries from the SDK.
#
# NOTE: on the IQ-8275 EVK you almost certainly DON'T need this. The board already ships the
# QNN runtime in /usr/lib (same 2.47 as the SDK), and those are the libs that actually load
# there — the SDK's x86-staged copies link against libc.so (a dev symlink the board lacks)
# and fail with "libc.so: cannot open shared object file". Keep this only for a board whose
# image has no QNN runtime; then you'd copy these next to the .bin AND make them resolve
# their own deps. For the EVK, skip it: copy just the context .bin over (see Step 7/8).
set +e
source "$(dirname "$0")/env.sh" || exit 1

OUT="${1:-runtime_libs}"
mkdir -p "$OUT"

# aarch64-* tolerates the exact triplet folder name varying across SDK versions
# (e.g. aarch64-oe-linux-gcc11.2, aarch64-ubuntu-gcc9.4). hexagon-v75 skel is the
# on-NPU kernel; the V75 stub + the core QNN libs are the aarch64 (CPU-side) halves.
cp "$SDK"/lib/aarch64-*/libQnnHtp.so                    "$OUT"/ 2>/dev/null
cp "$SDK"/lib/aarch64-*/libQnnSystem.so                 "$OUT"/ 2>/dev/null
cp "$SDK"/lib/aarch64-*/libQnnHtpNetRunExtensions.so    "$OUT"/ 2>/dev/null
cp "$SDK"/lib/aarch64-*/libQnnHtpPrepare.so             "$OUT"/ 2>/dev/null
cp "$SDK"/lib/aarch64-*/libQnnHtpV75Stub.so             "$OUT"/ 2>/dev/null
cp "$SDK"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so  "$OUT"/ 2>/dev/null

echo "===== staged into $OUT/ ====="
ls -la "$OUT"/
n=$(ls "$OUT"/*.so 2>/dev/null | wc -l)
echo "($n libraries)"
if [ "$n" -lt 6 ]; then
  echo "warn: expected 6 libraries — check your SDK's lib/aarch64-* and lib/hexagon-v75 folders." >&2
fi
echo
echo "Next: copy these + the context .bin + qnn-daemon-aarch64 to the board, e.g.:"
echo "  scp $OUT/*.so ctx16/best_a16w8_htpv75.bin daemon/qnn-daemon-aarch64 \\"
echo "      root@<board-ip>:/home/weston/npu/"
echo "===== END ====="
