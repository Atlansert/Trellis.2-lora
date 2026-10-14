"""
最简FastAPI应用入口
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import sys
import time

logger = logging.getLogger("seed_service")


# Always persist service and Uvicorn access logs next to this file.  The
# handlers are configured before Uvicorn starts so both direct and tmux-based
# launches write to the same log file without recording request payloads.
LOG_FILE = Path(__file__).with_name("seed_service.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")


# 创建FastAPI应用实例
app = FastAPI(
    title="3D模型生成服务",
    description="基于豆包 Seed3D 的3D模型生成API服务，接收单张侧视图生成GLB格式3D模型",
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    """记录请求元数据，不记录请求体中的图片 Base64。"""
    started = time.perf_counter()
    client = request.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown"
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception(
            "API request failed: ip=%s method=%s path=%s duration=%.3fs",
            client_addr, request.method, request.url.path, elapsed,
        )
        raise

    elapsed = time.perf_counter() - started
    logger.info(
        "API request: ip=%s method=%s path=%s status=%s duration=%.3fs",
        client_addr, request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.get("/")
async def root():
    """
    根路径
    """
    return JSONResponse(
        content={
            "message": "欢迎使用FastAPI",
            "status": "running",
            "docs": "/docs"
        }
    )


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return JSONResponse(
        content={
            "status": "healthy"
        }
    )


# 导入并注册3D模型生成路由
from model_generator import router as model_generator_router
app.include_router(model_generator_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8555,
        reload=False,
        log_config=None,
    )
