"""BiRefNet 抠图封装（rembg 实现）。

用 transformers 加载 BiRefNet 图像分割模型，预测前景蒙版，
缩回原图尺寸后写入 alpha 通道，返回带透明背景的 RGBA 图。
"""
from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    """BiRefNet 前景蒙版模型的最简封装。

    接口与 pipeline 约定的 rembg_model 一致：支持 to/cuda/cpu，
    调用时传入 PIL 图，返回同尺寸、alpha 为前景蒙版的 PIL 图。
    """
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # trust_remote_code：BiRefNet 的网络定义在 HF 仓库里而非 transformers 内置
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        # 预处理：缩放到 1024×1024 → ToTensor → ImageNet 均值方差归一化
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """预测前景蒙版并写入 alpha 通道，返回与原图同尺寸的 RGBA 图。"""
        image_size = image.size
        # 注意：输入固定放在 cuda 上推理（不随 to()/cpu() 设置变化）
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        # 蒙版缩回原图尺寸后作为 alpha 写回
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    