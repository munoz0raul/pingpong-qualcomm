#!/bin/bash
# Re-quantize YOLOv8n with 16-bit ACTIVATIONS (A16W8) to preserve the score, and
# regenerate the HTP V75 context binary. Reason: in pure INT8, box and score share
# the same scale; the coordinate range (0..580) crushes the score (0..1) to zero.
set +e
source /local/mnt/workspace/qairt/qairt_env.sh
SDK=/local/mnt/workspace/qairt/qairt/2.47.0.260601
cd /local/mnt/workspace/qairt || exit 1

echo "===== 1) Quantize A16W8 (act 16-bit, weights 8-bit) ====="
qairt-quantizer \
  --input_dlc /local/mnt/workspace/qairt/best_fp.dlc \
  --input_list /local/mnt/workspace/qairt/calib/input_list.txt \
  --act_bitwidth 16 \
  --weights_bitwidth 8 \
  --output_dlc /local/mnt/workspace/qairt/best_a16w8.dlc \
  2>&1 | grep -iE 'error|warning|quantiz|encoding|complete|bitwidth' | head -30
echo "rc quant: ${PIPESTATUS[0]}"
ls -la /local/mnt/workspace/qairt/best_a16w8.dlc 2>&1

echo
echo "===== 2) Regenerate context binary for HTP V75 ====="
# reuse the already-validated 2-file schema, just swap the graph_name for best_a16w8
cat > /local/mnt/workspace/qairt/htp_config.json <<'JSON'
{
  "graphs": [ { "graph_names": ["best_a16w8"], "vtcm_mb": 0, "O": 3 } ],
  "devices": [ { "htp_arch": "v75" } ]
}
JSON
cat > /local/mnt/workspace/qairt/backend_ext.json <<JSON
{
    "backend_extensions": {
        "shared_library_path": "$SDK/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
        "config_file_path": "/local/mnt/workspace/qairt/htp_config.json"
    }
}
JSON
rm -rf /local/mnt/workspace/qairt/ctx16 && mkdir -p /local/mnt/workspace/qairt/ctx16
qnn-context-binary-generator \
  --dlc_path /local/mnt/workspace/qairt/best_a16w8.dlc \
  --backend "$SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
  --output_dir /local/mnt/workspace/qairt/ctx16 \
  --binary_file best_a16w8_htpv75 \
  --config_file /local/mnt/workspace/qairt/backend_ext.json \
  2>&1 | grep -iE 'error|unknown key|v75|completed|serialized|graph' | head -30
echo "rc ctx: ${PIPESTATUS[0]}"
ls -la /local/mnt/workspace/qairt/ctx16/
echo "===== END ====="
