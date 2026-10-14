# 3D模型生成服务

基于豆包 Seed3D 的3D模型生成 REST API 服务，接收单张图片并生成 GLB 格式3D模型。

## 项目结构

```
3d_gen_online/
├── main.py              # FastAPI 应用入口，监听 8555 端口
├── model_generator.py   # 3D模型生成 REST API 路由
├── 3d_seed.py           # 豆包 Seed3D 独立批量调用脚本（需服务端 API Key）
├── call_seed3d.sh       # Linux/macOS/WSL 客户端脚本
├── requirements.txt     # Python 依赖
└── readme.md            # 项目文档
```

## 功能特点

- ✅ **RESTful API**: 基于 FastAPI 构建
- ✅ **3D模型生成**: 接收单张图片生成 GLB 模型
- ✅ **Base64支持**: 支持普通 Base64 或 Data URL 格式
- ✅ **豆包 Seed3D**: 使用 `doubao-seed3d-2-0-260328`
- ✅ **自动比例处理**: 自动将超出比例限制的图片白底补边
- ✅ **自动文档**: 提供 Swagger UI 和 OpenAPI JSON
- ✅ **服务端密钥**: `ARK_API_KEY` 只保存在服务端，不暴露给客户端

当前接口为单视图接口。同一个任务只接收一张图片；多张图片会创建多个独立任务，不会进行多视图联合重建。

## 快速开始

### 1. 创建环境并安装依赖

```bash
conda create -n doubao_seed python=3.10 pip
conda activate doubao_seed
pip install -r requirements.txt
```

### 2. 配置 API Key

在服务器环境中设置：

```bash
export ARK_API_KEY='your_ark_api_key'
```

不要将真实密钥写入代码、前端或提交到 Git。

### 3. 启动服务

```bash
cd /data/lzx/TRELLIS.2/3d_gen_online
conda activate doubao_seed
bash run.sh
```

服务监听：

```text
http://192.168.65.39:8555
```

`bash run.sh` 默认前台运行。服务会自动将 Uvicorn 访问日志、请求时间、调用方 IP/端口、HTTP 方法、路径、状态码、请求耗时、图片比例处理和异常写入 `seed_service.log`；不会记录图片 Base64 或 API Key。生产环境仍建议使用 `tmux`、`systemd` 或其他进程管理工具保持服务运行。

## API接口说明

### 1. 生成3D模型

**接口地址**：`POST /api/generate-3d-model`

**完整地址**：`http://192.168.65.39:8555/api/generate-3d-model`

**请求头**：

```http
Content-Type: application/json
```

**请求参数**：

```json
{
  "cst": "iVBORw0KGgoAAAANSUhEUg..."
}
```

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| cst | string | 是 | 单张图片的 Base64 字符串，可使用 PNG、JPEG 或 JPG |

`cst` 也可以使用 Data URL：

```text
data:image/png;base64,iVBORw0KGgo...
```

服务端会自动处理超出 Seed3D 支持范围 `0.40–2.50` 的图片宽高比。

**成功响应**：

```json
{
  "code": 200,
  "msg": "3D模型生成成功",
  "data": {
    "model_file": "Z2xURg...",
    "file_name": "generated_model.glb",
    "file_size": 123456,
    "file_format": "GLB",
    "encoding": "base64"
  }
}
```

客户端需要对 `data.model_file` 进行 Base64 解码，得到 GLB 二进制文件。

**失败响应**：

当 Seed3D 任务、结果下载或服务端发生异常时，接口返回 HTTP `500`，响应格式如下：

```json
{
  "code": 500,
  "msg": "3D模型生成失败: ...",
  "data": null
}
```

请求校验和配置错误返回 FastAPI 默认错误格式，不包含 `code/msg/data` 字段：

```json
{
  "detail": "错误说明"
}
```

常见 HTTP 状态码：

- `400`：请求中的 Base64 无效、Data URL 格式错误，或内容不是有效图片；
- `422`：请求体校验失败，例如缺少 `cst` 字段或 `cst` 不是字符串；
- `503`：服务端未安装 Seed3D SDK，或未配置 `ARK_API_KEY`；
- `500`：Seed3D 任务失败、结果下载失败或其他服务端异常。

生成任务可能需要较长时间。客户端 HTTP 超时应设置为至少 `1900` 秒，并在等待期间保持连接。

### 前端跨域说明

当前 FastAPI 服务未启用 CORS。浏览器前端通过不同域名、协议或端口直接调用时，可能出现跨域错误。建议采用以下任一方式：

- 前端开发服务器配置反向代理，使请求与前端保持同源；
- 由部署人员在 FastAPI 前增加 CORS 中间件或在 Nginx 网关配置 CORS；
- 前端与 API 部署在同一源下。

跨域配置不影响 Linux、Python 或其他非浏览器客户端。

### 2. 获取服务信息

**接口地址**：`GET /api/model/info`

```bash
curl http://192.168.65.39:8555/api/model/info
```

**返回示例**：

```json
{
  "code": 200,
  "msg": "获取服务信息成功",
  "data": {
    "service": "3D模型生成服务",
    "version": "1.1.0",
    "model": "doubao-seed3d-2-0-260328",
    "supported_formats": ["GLB"],
    "input_requirements": {
      "image_count": 1,
      "image_format": "单张侧视图的base64编码",
      "supported_image_types": ["PNG", "JPEG", "JPG"]
    },
    "output_format": "GLB (GL Transmission Format Binary)",
    "status": "running"
  }
}
```

### 3. 健康检查

**接口地址**：`GET /health`

```bash
curl http://192.168.65.39:8555/health
```

**返回示例**：

```json
{"status":"healthy"}
```

### 4. 接口调用约定

- 生成接口一次只接收一张图片；
- 请求体必须是 `application/json`，字段名必须为 `cst`；
- `cst` 只传图片 Base64，不要传本地文件路径；
- 成功时 HTTP 状态码为 `200`，并从 `data.model_file` Base64 解码 GLB；
- 生成失败时，HTTP 状态码与 JSON 内容中的 `code` 通常一致（`500`）；
- `400`、`422`、`503` 等 FastAPI 校验或配置错误使用 `{"detail":"错误说明"}` 格式；
- 生成接口是同步等待式调用，服务端会在同一 HTTP 请求中等待豆包任务完成；
- 客户端应处理网络超时、HTTP `400/503/500` 以及 JSON 中的 `code != 200`。

## 使用示例

### Python调用示例

```python
import base64
import requests

with open("side_view.png", "rb") as f:
    payload = {"cst": base64.b64encode(f.read()).decode("ascii")}

response = requests.post(
    "http://192.168.65.39:8555/api/generate-3d-model",
    json=payload,
    timeout=1900,
)
result = response.json()

if result["code"] != 200:
    raise RuntimeError(result["msg"])

with open("output.glb", "wb") as f:
    f.write(base64.b64decode(result["data"]["model_file"]))
```

### cURL调用示例（Linux/macOS）

由于图片 Base64 可能很长，应先写入 JSON 文件：

```bash
python3 -c 'import base64,json,pathlib; p=pathlib.Path("side_view.png"); pathlib.Path("request.json").write_text(json.dumps({"cst":base64.b64encode(p.read_bytes()).decode()}))'
curl --max-time 1900 -X POST \
  "http://192.168.65.39:8555/api/generate-3d-model" \
  -H "Content-Type: application/json" \
  --data-binary @request.json -o response.json
python3 -c 'import base64,json,pathlib; r=json.loads(pathlib.Path("response.json").read_text()); pathlib.Path("output.glb").write_bytes(base64.b64decode(r["data"]["model_file"]))'
```

### Linux脚本调用

```bash
chmod +x call_seed3d.sh
./call_seed3d.sh ./side_view.png ./output.glb
```

`3d_seed.py` 是直接调用豆包 SDK 的独立脚本，不经过本 REST 服务；运行它时需要在当前环境设置 `ARK_API_KEY`，并建议显式指定 `--fileformat glb`。


## 当前实现

- ✅ 接收单张图片 Base64
- ✅ 验证图片 Base64 和图片格式
- ✅ 自动处理超宽或超高图片
- ✅ 调用豆包 Seed3D 创建任务并轮询结果
- ✅ 支持 GLB 或 ZIP 中的 GLB 返回文件
- ✅ 返回包含 GLB Base64 的 JSON
- ✅ 完整的异常处理

## 常用命令

```bash
# 启动服务
conda activate doubao_seed
cd /data/lzx/TRELLIS.2/3d_gen_online
bash run.sh

# 使用 tmux 后台运行
tmux new-session -d -s doubao_seed \
  'source /home/test/miniconda3/envs/doubao_seed/etc/conda/activate.d/ark_api_key.sh && python main.py'

# 查看服务状态
curl http://192.168.65.39:8555/health

# 查看 tmux 日志
tmux capture-pane -pt doubao_seed -S -100

# 查看持久化服务日志（推荐）
tail -f /data/lzx/TRELLIS.2/3d_gen_online/seed_service.log

# 使用 nohup 时查看重定向日志（示例）
# nohup bash run.sh > seed_service.log 2>&1 &
# tail -f seed_service.log

# 查看进程和端口
ps aux | grep "python main.py"
ss -ltnp | grep :8555

# 停止 tmux 服务
tmux kill-session -t doubao_seed

# 查看自动生成的 API 文档
# 浏览器访问：http://192.168.65.39:8555/docs
```
