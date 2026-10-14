import base64
import subprocess
import json

# 读取并编码图片
with open("assets/car_assets/black_car_front.png", "rb") as f:
    img_bytes = f.read()
img_base64_front = base64.b64encode(img_bytes).decode("utf-8")

with open("assets/car_assets/black_car_side.png", "rb") as f:
    img_bytes = f.read()
img_base64_side = base64.b64encode(img_bytes).decode("utf-8")

# 构造 JSON 数据
payload = {
    "zst": img_base64_front,
    "cst": img_base64_side
}
payload_str = json.dumps(payload)

# 通过 stdin 传给 curl
subprocess.run([
    "curl",
    "-X", "POST",
    "http://localhost:18000/api/generate-3d-model",
    "-H", "Content-Type: application/json",
    "--data-binary", "@-"
], input=payload_str.encode("utf-8"))
