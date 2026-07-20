#!/bin/bash
# Re-quantize YOLOv8n with 16-bit ACTIVATIONS (A16W8) to preserve the score, and
# regenerate the HTP V75 context binary. Reason: in pure INT8, box and score share
# the same scale; the coordinate range (0..580) crushes the score (0..1) to zero.
set +e
source "$(dirname "$0")/env.sh" || exit 1
cd "$WORK" || exit 1

echo "===== 1) Quantize A16W8 (act 16-bit, weights 8-bit) ====="
qairt-quantizer \
  --input_dlc "$WORK/best_fp.dlc" \
  --input_list "$WORK/calib/input_list.txt" \
  --act_bitwidth 16 \
  --weights_bitwidth 8 \
  --output_dlc "$WORK/best_a16w8.dlc" \
  2>&1 | grep -iE 'error|warning|quantiz|encoding|complete|bitwidth' | head -30
echo "rc quant: ${PIPESTATUS[0]}"
ls -la "$WORK/best_a16w8.dlc" 2>&1

echo
echo "===== 2) Regenerate context binary for HTP V75 ====="
# reuse the already-validated 2-file schema, just swap the graph_name for best_a16w8
cat > "$WORK/htp_config.json" <<'JSON'
{
  "graphs": [ { "graph_names": ["best_a16w8"], "vtcm_mb": 0, "O": 3 } ],
  "devices": [ { "htp_arch": "v75" } ]
}
JSON
cat > "$WORK/backend_ext.json" <<JSON
{
    "backend_extensions": {
        "shared_library_path": "$SDK/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
        "config_file_path": "$WORK/htp_config.json"
    }
}
JSON
rm -rf "$WORK/ctx16" && mkdir -p "$WORK/ctx16"
qnn-context-binary-generator \
  --dlc_path "$WORK/best_a16w8.dlc" \
  --backend "$SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
  --output_dir "$WORK/ctx16" \
  --binary_file best_a16w8_htpv75 \
  --config_file "$WORK/backend_ext.json" \
  2>&1 | grep -iE 'error|unknown key|v75|completed|serialized|graph' | head -30
echo "rc ctx: ${PIPESTATUS[0]}"
ls -la "$WORK/ctx16/"
echo "===== END ====="
