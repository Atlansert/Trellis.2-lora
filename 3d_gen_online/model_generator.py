"""
3D模型生成模块
接收单张侧视图的base64编码，通过豆包 Seed3D 生成GLB格式3D模型
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
import base64
import os
from PIL import Image
import io
import traceback
import time
import urllib.request
import zipfile
import logging

logger = logging.getLogger("seed_service")

# Seed3D 配置。密钥只从环境变量读取，禁止写入源码。
SEED3D_MODEL = "doubao-seed3d-2-0-260328"
SEED3D_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SEED3D_MESH_QUALITY = "high"
SEED3D_POLL_INTERVAL = 10
SEED3D_TIMEOUT = 1800
SEED3D_MIN_ASPECT_RATIO = 0.40
SEED3D_MAX_ASPECT_RATIO = 2.50
SEED3D_MAX_IMAGE_WIDTH = 4096

# 创建路由器
router = APIRouter(
    prefix="/api",
    tags=["3D模型生成"],
)


class Image3DRequest(BaseModel):
    """
    3D模型生成请求模型
    """
    cst: str = Field(..., description="侧视图图片的base64编码字符串")

    class Config:
        json_schema_extra = {
            "example": {
                "cst": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
        }


class ApiResponse(BaseModel):
    """
    统一API响应模型
    """
    code: int = Field(..., description="响应码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")


def _prepare_image_for_seed3d(image: Image.Image) -> Image.Image:
    """将图片调整到 Seed3D 支持的尺寸和宽高比，尽量保留完整主体。"""
    image = image.convert("RGBA")
    width, height = image.size

    if width > SEED3D_MAX_IMAGE_WIDTH:
        scale = SEED3D_MAX_IMAGE_WIDTH / width
        width = SEED3D_MAX_IMAGE_WIDTH
        height = max(1, int(round(height * scale)))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    ratio = width / height
    if SEED3D_MIN_ASPECT_RATIO <= ratio <= SEED3D_MAX_ASPECT_RATIO:
        return image

    if ratio > SEED3D_MAX_ASPECT_RATIO:
        new_height = int((width / SEED3D_MAX_ASPECT_RATIO) + 0.999999)
        new_width = width
    else:
        new_width = int((height * SEED3D_MIN_ASPECT_RATIO) + 0.999999)
        new_height = height

    canvas = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 255))
    offset = ((new_width - width) // 2, (new_height - height) // 2)
    canvas.alpha_composite(image, offset)
    logger.info(
        "Image padded from %sx%s (ratio %.2f) to %sx%s",
        width, height, ratio, new_width, new_height,
    )
    return canvas


def generate_glb_with_seed3d(image: Image.Image) -> bytes:
    """传入一张 PIL 图片，调用豆包 Seed3D 并返回 GLB 二进制。"""
    try:
        from volcenginesdkarkruntime import Ark
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Seed3D SDK 未安装，请安装 volcengine-python-sdk[ark]",
        ) from exc

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="服务端未配置 ARK_API_KEY")

    # Seed3D 接收 image_url。统一编码为 PNG data URL，调用方只需要传图片。
    image_buffer = io.BytesIO()
    _prepare_image_for_seed3d(image).save(image_buffer, format="PNG")
    image_data_url = "data:image/png;base64," + base64.b64encode(
        image_buffer.getvalue()
    ).decode("ascii")

    client = Ark(base_url=SEED3D_BASE_URL, api_key=api_key)
    task = client.content_generation.tasks.create(
        model=SEED3D_MODEL,
        content=[
            {"type": "text", "text": f"--meshquality {SEED3D_MESH_QUALITY} --fileformat glb"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    )

    deadline = time.time() + SEED3D_TIMEOUT
    while time.time() < deadline:
        result = client.content_generation.tasks.get(task_id=task.id)
        status = getattr(result, "status", None)
        if status == "succeeded":
            file_url = getattr(getattr(result, "content", None), "file_url", None)
            if not file_url:
                raise RuntimeError("Seed3D 任务成功但未返回 file_url")
            with urllib.request.urlopen(file_url, timeout=300) as response:
                payload = response.read()

            # Seed3D 可能直接返回 GLB，也可能返回包含 GLB 的 ZIP。
            if payload[:4] == b"glTF":
                return payload
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    glb_members = [name for name in archive.namelist() if name.lower().endswith(".glb")]
                    if not glb_members:
                        raise RuntimeError("Seed3D 返回的压缩包中未找到 GLB 文件")
                    return archive.read(glb_members[0])
            except zipfile.BadZipFile as exc:
                raise RuntimeError("Seed3D 返回的文件既不是 GLB 也不是有效 ZIP") from exc
        if status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"Seed3D 任务失败（{status}）：{getattr(result, 'error', result)}")
        time.sleep(SEED3D_POLL_INTERVAL)

    raise TimeoutError(f"Seed3D 任务 {task.id} 在 {SEED3D_TIMEOUT} 秒内未完成")


@router.post("/generate-3d-model", summary="3D文件生成")
async def generate_3d_model(request: Image3DRequest):
    """
    3D模型生成接口

    接收单张侧视图的base64编码，通过豆包 Seed3D 生成3D模型文件

    参数:
    - cst: 侧视图图片的base64编码字符串

    返回:
    - GLB格式的3D模型文件（二进制流）

    示例请求体:
    ```json
    {
        "cst": "data:image/png;base64,iVBORw0KGgo..."
    }
    ```
    """
    start_time=time.time()

    try:
        # ============================================================
        # 第一步：解析和验证输入的图片base64字符串
        # ============================================================

        # 如果base64字符串包含data URL前缀，需要去除
        cst_base64 = request.cst
        if cst_base64.startswith('data:'):
            header, separator, cst_base64 = cst_base64.partition(',')
            if not separator or not header.endswith(';base64'):
                raise HTTPException(
                    status_code=400,
                    detail="图片 data URL 格式无效，应为 data:<mime>;base64,<content>",
                )

        # 验证base64字符串的有效性
        try:
            decoded_cst = base64.b64decode(cst_base64, validate=True)
            cst_image = Image.open(io.BytesIO(decoded_cst)).convert("RGBA")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Base64解码失败: {str(e)}"
            )

        # ============================================================
        # 第二步：调用豆包 Seed3D 生成GLB
        # ============================================================

        glb_data = generate_glb_with_seed3d(cst_image)
        # ============================================================
        # 第三步：返回生成的3D模型文件
        # ============================================================

        # 将GLB二进制数据转换为base64编码
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')

        # 返回标准JSON响应
        end_time=time.time()
        logger.info("Request completed in %.3f seconds", end_time - start_time)
        return JSONResponse(
            content={
                "code": 200,
                "msg": "3D模型生成成功",
                "data": {
                    "model_file": glb_base64,
                    "file_name": "generated_model.glb",
                    "file_size": len(glb_data),
                    "file_format": "GLB",
                    "encoding": "base64"
                }            }
        )

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 捕获其他异常并返回500错误
        end_time=time.time()
        logger.exception("Request failed after %.3f seconds", end_time - start_time)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "msg": f"3D模型生成失败: {str(e)}",
                "data": None
            }
        )


@router.get("/model/info", summary="获取3D模型信息")
async def get_model_info():
    """
    获取3D模型生成服务的信息

    返回当前服务支持的功能和状态
    """
    return JSONResponse(
        content={
            "code": 200,
            "msg": "获取服务信息成功",
            "data": {
                "service": "3D模型生成服务",
                "version": "1.1.0",
                "model": SEED3D_MODEL,
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
    )
