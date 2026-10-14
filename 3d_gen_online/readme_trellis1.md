# 3D模型生成服务

基于大模型的3D模型生成API服务，接收两张图片生成GLB格式3D模型。

## 项目结构

```
crrc-3d-model/
├── main.py              # 应用入口文件
├── model_generator.py   # 3D模型生成路由模块
├── black_car.glb        # 示例3D模型文件
├── requirements.txt     # 项目依赖
└── README.md           # 项目文档
```

## 功能特点

- ✅ **RESTful API**: 基于FastAPI构建的高性能API
- ✅ **3D模型生成**: 接收两张图片生成GLB格式3D模型
- ✅ **Base64支持**: 支持图片的base64编码输入
- ✅ **二进制流返回**: 直接返回GLB文件二进制数据
- ✅ **自动文档**: 完整的Swagger UI文档
- ✅ **零依赖配置**: 无需数据库、Redis等外部依赖

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装核心依赖:

```bash
pip install fastapi uvicorn[standard] requests
```

### 2. 启动服务

**使用 nohup 后台运行（生产环境）**

```bash
# 后台运行，日志输出到 nohup.out
nohup python main.py > nohup.out 2>&1 &

```

**查看和管理后台进程：**

```bash
# 查看日志
tail -f nohup.out

# 查找运行中的进程
ps aux | grep "python main.py"

# 停止服务（使用进程ID）
kill <PID>

# 或者强制停止
pkill -f "python main.py"
```


## API接口说明

### 1. 生成3D模型

**接口地址**: `POST /api/generate-3d-model`

**请求参数**:

```json
{
  "zst": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...",
  "cst": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ..."
}
```

| 参数  | 类型 | 必填 | 说明 |
|-----|------|------|------|
| zst | string | 是 | 正视图图片的base64编码字符串 |
| cst | string | 是 | 侧视图图片的base64编码字符串 |

**返回**: GLB格式3D模型文件（二进制流）

**响应头**:
- `Content-Type: model/gltf-binary`
- `Content-Disposition: attachment; filename=generated_model.glb`

### 2. 获取服务信息

**接口地址**: `GET /api/model/info`

**返回示例**:

```json
{
  "service": "3D模型生成服务",
  "version": "1.0.0",
  "supported_formats": ["GLB"],
  "input_requirements": {
    "image_count": 2,
    "image_format": "base64编码的图片",
    "supported_image_types": ["PNG", "JPEG", "JPG"]
  },
  "output_format": "GLB (GL Transmission Format Binary)",
  "status": "running"
}
```

## 使用示例

### Python调用示例

```python
import base64
import requests

# 将图片转换为base64
def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求数据
payload = {
    "zst": image_to_base64("front_view.png"),  # 正视图
    "cst": image_to_base64("side_view.png")     # 侧视图
}

# 发送请求
response = requests.post(
    "http://localhost:8000/api/generate-3d-model",
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
```

### cURL调用示例

```bash
curl -X POST "http://localhost:8000/api/generate-3d-model" \
  -H "Content-Type: application/json" \
  -d '{
    "zst": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...",
    "cst": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ..."
  }'
```


### 当前实现

- ✅ 接收两张图片的base64编码
- ✅ 验证base64数据有效性
- ✅ 返回示例GLB文件（black_car.glb）
- ✅ 完整的错误处理和异常捕获

### 大模型集成（待实现）

在 `model_generator.py` 中已经预留了大模型推理的位置，包含详细的TODO注释，需要实现:

```python
# TODO: 大模型推理部分
# 实际实现时，需要调用大模型API进行3D模型生成
#
# 伪代码示例：
# model_service = Model3DService()
# glb_data = await model_service.generate_from_images(
#     zst=decoded_zst,  # 正视图
#     cst=decoded_cst,  # 侧视图
#     model_name="triposr",
#     parameters={
#         "quality": "high",
#         "texture_resolution": 1024
#     }
# )
```

## 常用命令

```bash
# ========== 服务启动 ==========

# 开发环境启动（支持热重载）
python main.py

# 使用uvicorn启动
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境启动（后台运行）
nohup python main.py > nohup.out 2>&1 &

# 指定端口启动
nohup python -c "uvicorn main:app --host 0.0.0.0 --port 18000" > nohup.out 2>&1 &

# ========== 进程管理 ==========

# 查看所有Python进程
ps aux | grep python

# 查看特定应用的进程
ps aux | grep "python main.py"

# 查看进程端口占用
netstat -tulnp | grep :8080
# 或者
lsof -i :8000

# 查看日志（实时）
tail -f nohup.out

# 查看最后100行日志
tail -n 100 nohup.out

# 停止服务（使用进程ID）
kill <PID>

# 强制停止服务
kill -9 <PID>

# 停止所有相关进程
pkill -f "python main.py"
pkill -f "uvicorn main:app"

# ========== 测试命令 ==========

# 运行测试脚本
python test_api.py

# 手动测试API（使用curl）
curl -X POST http://localhost:18000/api/generate-3d-model \
  -H "Content-Type: application/json" \
  -d '{"zst":"iVBORw0KGgo...","cst":"iVBORw0KGgo..."}'

# 获取服务信息
curl http://localhost:18000/api/model/info

# 查看API文档
# 浏览器访问: http://localhost:8000/docs

# ========== 依赖管理 ==========

# 安装依赖
pip install -r requirements.txt

# 导出当前环境依赖
pip freeze > requirements.txt

# 查看已安装的包
pip list
```
