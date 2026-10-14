#!/usr/bin/env bash
# Seed3D image-to-GLB client
# Usage:
#   ./call_seed3d.sh input.png [output.glb]
# Examples:
#   ./call_seed3d.sh ./001.png
#   ./call_seed3d.sh ./001.png ./001.glb
# Optional server override:
#   SERVER=http://192.168.65.39:8555 ./call_seed3d.sh input.png

set -euo pipefail
IMAGE_PATH="${1:?Usage: $0 input.png [output.glb]}"
OUTPUT_PATH="${2:-${IMAGE_PATH%.*}_generated.glb}"
SERVER="${SERVER:-http://192.168.65.39:8555}"

[[ -f "$IMAGE_PATH" ]] || { echo "Image not found: $IMAGE_PATH" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 - "$IMAGE_PATH" "$tmp/request.json" <<'PY'
import base64, json, pathlib, sys
image, request = sys.argv[1:]
data = base64.b64encode(pathlib.Path(image).read_bytes()).decode("ascii")
pathlib.Path(request).write_text(json.dumps({"cst": data}, separators=(",", ":")))
PY

echo "Calling Seed3D; please wait..."
curl -fsS --max-time 1900 "$SERVER/api/generate-3d-model" \
  -H 'Content-Type: application/json' \
  --data-binary "@$tmp/request.json" -o "$tmp/response.json"

python3 - "$tmp/response.json" "$OUTPUT_PATH" <<'PY'
import base64, json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
if r.get("code") != 200 or not r.get("data", {}).get("model_file"):
    raise SystemExit("Seed3D failed: " + str(r.get("msg", r)))
pathlib.Path(sys.argv[2]).write_bytes(base64.b64decode(r["data"]["model_file"]))
print("Generated: " + sys.argv[2])
PY
