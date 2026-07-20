#!/bin/bash
# 1) Inspect the ONNX I/O. 2) Convert ONNX -> DLC (float) with qairt-converter.
set +e
source "$(dirname "$0")/env.sh" || exit 1
cd "$WORK" || exit 1

echo "===== best.onnx I/O (names and shapes) ====="
python3 - <<PY
import onnx
m = onnx.load("$WORK/best.onnx")
g = m.graph
def shp(t):
    return [d.dim_value if d.dim_value>0 else d.dim_param for d in t.type.tensor_type.shape.dim]
print("INPUTS:")
for i in g.input:
    print(f"  {i.name}  {shp(i)}")
print("OUTPUTS:")
for o in g.output:
    print(f"  {o.name}  {shp(o)}")
PY

echo
echo "===== qairt-converter: ONNX -> DLC (float, not quantized yet) ====="
qairt-converter \
  --input_network "$WORK/best.onnx" \
  --output_path "$WORK/best_fp.dlc" \
  2>&1 | tail -40
echo "rc=$?"

echo
echo "===== DLC generated? ====="
ls -la "$WORK/best_fp.dlc" 2>&1

echo "===== END ====="
