#!/bin/bash
# Generate the HTP V75 context binary with the CORRECT SCHEMA (2 files: backend_ext + htp config).
set +e
source "$(dirname "$0")/env.sh" || exit 1
cd "$WORK" || exit 1
rm -rf ctx && mkdir -p ctx

# 1) HTP internal config: graph + V75 arch (the board's QCS8300 NPU)
cat > "$WORK/htp_config.json" <<'JSON'
{
  "graphs": [
    {
      "graph_names": ["best_int8"],
      "vtcm_mb": 0,
      "O": 3
    }
  ],
  "devices": [
    {
      "htp_arch": "v75"
    }
  ]
}
JSON

# 2) backend extensions: points to the lib and to the internal config (ABSOLUTE path)
cat > "$WORK/backend_ext.json" <<JSON
{
    "backend_extensions": {
        "shared_library_path": "$SDK/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
        "config_file_path": "$WORK/htp_config.json"
    }
}
JSON

echo "===== htp_config.json ====="; cat "$WORK/htp_config.json"
echo "===== backend_ext.json ====="; cat "$WORK/backend_ext.json"

echo
echo "===== Generate context binary (correct config) ====="
qnn-context-binary-generator \
  --dlc_path "$WORK/best_int8.dlc" \
  --backend "$SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
  --output_dir "$WORK/ctx" \
  --binary_file best_int8_htpv75 \
  --config_file "$WORK/backend_ext.json" \
  2>&1 | grep -iE 'error|unknown key|htp_arch|v75|stage|completed|serialized|graph' | head -40
echo "rc(grep pipe)"

echo
echo "===== Result ====="
ls -la "$WORK/ctx/"
echo "===== END ====="
