#!/bin/bash
# Run the A16W8 YOLO (act 16-bit) on the HTP V75 NPU — version that preserves the score.
# Runs ON THE BOARD (aarch64). The board already ships the QNN runtime in /usr/bin +
# /usr/lib (same 2.47 as the SDK) — we do NOT copy any .so from the x86 box. NPU_DIR only
# holds the context .bin + the test input (default /home/weston/npu).
set +e
NPU_DIR="${NPU_DIR:-/home/weston/npu}"
cd "$NPU_DIR" || exit 1

echo "$NPU_DIR/emeet2_input.raw" > "$NPU_DIR/input_list.txt"

cat > "$NPU_DIR/htp_config.json" <<'JSON'
{
  "graphs": [ { "graph_names": ["best_a16w8"], "vtcm_mb": 0 } ],
  "devices": [ { "htp_arch": "v75" } ]
}
JSON
cat > "$NPU_DIR/backend_ext.json" <<JSON
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "$NPU_DIR/htp_config.json"
  }
}
JSON

rm -rf "$NPU_DIR/out16" && mkdir -p "$NPU_DIR/out16"
echo "===== qnn-net-run A16W8 on the NPU ====="
qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context "$NPU_DIR/best_a16w8_htpv75.bin" \
  --input_list "$NPU_DIR/input_list.txt" \
  --config_file "$NPU_DIR/backend_ext.json" \
  --output_dir "$NPU_DIR/out16" \
  2>&1 | tail -20
echo "rc=$?"
echo "===== Outputs ====="
find "$NPU_DIR/out16" -type f 2>/dev/null

# qnn-net-run writes out16/Result_0/output0.raw; decode_npu_out.py reads npu_out.raw
# next to itself. Bridge the two so the decode step is a plain `python3 decode_npu_out.py`.
OUT_RAW="$NPU_DIR/out16/Result_0/output0.raw"
if [ -f "$OUT_RAW" ]; then
  cp "$OUT_RAW" "$NPU_DIR/npu_out.raw"
  echo "copied -> $NPU_DIR/npu_out.raw"
fi
echo "===== END ====="
