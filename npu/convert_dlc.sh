#!/bin/bash
# 1) Inspect the ONNX I/O. 2) Convert ONNX -> DLC (float) with qairt-converter.
set +e
source /local/mnt/workspace/qairt/qairt_env.sh
cd /local/mnt/workspace/qairt || exit 1

echo "===== best.onnx I/O (names and shapes) ====="
python3 - <<'PY'
import onnx
m = onnx.load("/local/mnt/workspace/qairt/best.onnx")
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
  --input_network /local/mnt/workspace/qairt/best.onnx \
  --output_path /local/mnt/workspace/qairt/best_fp.dlc \
  2>&1 | tail -40
echo "rc=$?"

echo
echo "===== DLC generated? ====="
ls -la /local/mnt/workspace/qairt/best_fp.dlc 2>&1

echo "===== END ====="
