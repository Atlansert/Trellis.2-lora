"""图像条件 mixin 与 DINO 特征提取器。

职责：用 DINOv2/DINOv3 提取单图或多视角图像特征作为 flow matching 的条件；
提供训练（可 CFG 丢弃）与推理两套条件接口，以及条件可视化。
输入：BCHW 的 float tensor（已归一化前）或 PIL.Image 列表；输出：patch token 特征 (B, N, D)。
契约：负条件统一用同形状全零特征（表示"无图像条件"）；多视角版按每样本视角数变长，
由上层 CFG mixin 负责按样本丢弃，提取器本身在 no_grad 下运行、权重转 bf16 放在当前 GPU。
"""

from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel, Dinov2Model
import numpy as np
from PIL import Image

from ....utils import dist_utils


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    DINOv2 特征提取器（torch.hub 拉取预训练权重），返回 layer_norm 后的 patch token。
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        # PIL 输入：缩放到该提取器约定的尺寸，转 RGB、[0,1] 归一化后组 batch 上 GPU
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']    # 取最后一层未归一化特征
        patchtokens = F.layer_norm(features, features.shape[-1:])      # 仅按最后一维做 LayerNorm
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    DINOv3 特征提取器（HF from_pretrained 加载），返回 layer_norm 后的 patch token。
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        # 手动逐层跑 ViT 主干：先 patch embed + RoPE 位置编码，再顺序过每个 Transformer 层
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        # PIL 输入：缩放到该提取器约定的尺寸，转 RGB、[0,1] 归一化后组 batch 上 GPU
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features

class DinoV2LocalFeatureExtractor:
    """Local DINOv2 extractor matching TRELLIS.1 preprocessing without HF download."""
    # 本地权重的 DINOv2 提取器：local_files_only 避免联网，预处理与 TRELLIS.1 对齐；
    # 只接受 BCHW tensor 输入，输出同样做最后一维 LayerNorm。
    def __init__(self, model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518):
        self.model = Dinov2Model.from_pretrained(model_name, local_files_only=True).eval()
        self.image_size=image_size
        self.transform=transforms.Compose([transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    def to(self, device): self.model.to(device)
    def cuda(self): self.model.cuda()
    def cpu(self): self.model.cpu()
    @torch.no_grad()
    def __call__(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4: raise ValueError('image must be BCHW tensor')
        image=self.transform(image).cuda()
        out=self.model(pixel_values=image)
        return F.layer_norm(out.last_hidden_state, out.last_hidden_state.shape[-1:])
    

class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    单图条件 mixin：把一张输入图编码成 DINO 特征作为条件。
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        惰性初始化：按配置里的 name 从本模块 globals 取提取器类，用 args 实例化后转 GPU+bf16。
        """
        with dist_utils.local_master_first():
            self.image_cond_model = globals()[self.image_cond_model_config['name']](**self.image_cond_model_config.get('args', {}))
            # Cast to GPU + bf16 in one call to avoid device/dtype mismatch under AMP.
            self.image_cond_model.model = self.image_cond_model.model.to(
                device=torch.cuda.current_device(), dtype=torch.bfloat16
            )

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        单图：首次调用时才加载提取器，no_grad 前向得到条件特征 (B, N, D)。
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        features = self.image_cond_model(image)
        return features
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        训练用：编码图像，负条件 = 同形状全零特征，再交给上层 CFG mixin 随机丢弃。
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        推理用：同上但不丢弃条件，cond/neg_cond 一起交给采样器做 CFG 引导。
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        直接返回图像 tensor 供 trainer 记录（type='image' 表示按图像写日志）。
        """
        return {'image': {'value': cond, 'type': 'image'}}
    

class MultiImageConditionedMixin:
    """
    Mixin for multiple-image-conditioned models.
    
    多视角条件 mixin：每个样本可带若干张视角图，特征按样本拼成变长列表。
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        惰性初始化：按配置里的 name 从本模块 globals 取提取器类，用 args 实例化后转 GPU+bf16。
        """
        with dist_utils.local_master_first():
            self.image_cond_model = globals()[self.image_cond_model_config['name']](**self.image_cond_model_config.get('args', {}))
            # Cast to GPU + bf16 in one call to avoid device/dtype mismatch under AMP.
            self.image_cond_model.model = self.image_cond_model.model.to(
                device=torch.cuda.current_device(), dtype=torch.bfloat16
            )

    @torch.no_grad()
    def encode_images(self, images: Union[List[torch.Tensor], List[List[Image.Image]]]) -> List[torch.Tensor]:
        """
        Encode the image.
        多视角：记录每样本视角数 -> 所有视角拼成一个 batch 一次前向 -> 按视角数切回，
        每个样本再 reshape 成 (视角数*patch数, D) 的二维条件。
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        seqlen = [len(i) for i in images]    # 每个样本的视角数
        images = torch.cat(images, dim=0) if isinstance(images[0], torch.Tensor) else sum(images, [])
        features = self.image_cond_model(images)
        features = torch.split(features, seqlen)
        features = [feature.reshape(-1, feature.shape[-1]) for feature in features]
        return features
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        训练用：编码多视角图像；负条件为每个样本一个全零 token（长度 1），表示"无条件"。
        """
        cond = self.encode_images(cond)
        # 注意：这里负数条件形状为 (1, D)，与正的变长条件不同，由上层 mixin/模型自行处理
        kwargs['neg_cond'] = [
            torch.zeros_like(cond[0][:1, :]) for _ in range(len(cond))
        ]
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        推理用：同上（负条件为单 token 全零），不做条件丢弃。
        """
        cond = self.encode_images(cond)
        kwargs['neg_cond'] = [
            torch.zeros_like(cond[0][:1, :]) for _ in range(len(cond))
        ]
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        每个样本最多取 4 张视角图，拼成 2x2 画布用于训练日志预览。
        """
        H, W = cond[0].shape[-2:]
        vis = []
        for images in cond:
            canvas = torch.zeros(3, H * 2, W * 2, device=images.device, dtype=images.dtype)
            for i, image in enumerate(images):
                if i == 4:
                    break
                kh = i // 2
                kw = i % 2
                canvas[:, kh*H:(kh+1)*H, kw*W:(kw+1)*W] = image
            vis.append(canvas)
        vis = torch.stack(vis)
        return {'image': {'value': vis, 'type': 'image'}}
