import base64
import requests

# 将图片转换为base64
def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求数据
payload = {
    "cst": image_to_base64("assets/car_assets/black_car_side.png")  # 侧视图
}

# 发送请求
response = requests.post(
    "http://192.168.65.39:8555/api/generate-3d-model",
    json=payload
)

# 解析响应
result = response.json()
if result['code'] == 200:
    # 获取base64编码的GLB文件
    import base64
    glb_data = base64.b64decode(result['data']['model_file'])

    # 保存文件
    with open("output.glb", 'wb') as f:
        f.write(glb_data)
    print("3D模型生成成功!")
