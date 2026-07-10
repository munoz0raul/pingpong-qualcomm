#!/bin/bash
# Run the A16W8 YOLO (act 16-bit) on the HTP V75 NPU — version that preserves the score.
set +e
cd /home/weston/npu || exit 1

echo "/home/weston/npu/emeet2_input.raw" > /home/weston/npu/input_list.txt

cat > /home/weston/npu/htp_config.json <<'JSON'
{
  "graphs": [ { "graph_names": ["best_a16w8"], "vtcm_mb": 0 } ],
  "devices": [ { "htp_arch": "v75" } ]
}
JSON
cat > /home/weston/npu/backend_ext.json <<'JSON'
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "/home/weston/npu/htp_config.json"
  }
}
JSON

rm -rf /home/weston/npu/out16 && mkdir -p /home/weston/npu/out16
echo "===== qnn-net-run A16W8 on the NPU ====="
qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context /home/weston/npu/best_a16w8_htpv75.bin \
  --input_list /home/weston/npu/input_list.txt \
  --config_file /home/weston/npu/backend_ext.json \
  --output_dir /home/weston/npu/out16 \
  2>&1 | tail -20
echo "rc=$?"
echo "===== Outputs ====="
find /home/weston/npu/out16 -type f 2>/dev/null
echo "===== END ====="
