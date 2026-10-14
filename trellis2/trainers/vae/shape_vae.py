"""形状 VAE 训练器（结构化潜变量 / 稀疏体素）。

职责：训练形状 VAE——encoder 把每个活跃体素的顶点与相交标记编码为潜变量，
decoder 重建网格（顶点、相交标记、细分结果）；除直接回归外还用随机相机渲染
GT 与重建网格，用 mask/depth/normal 的 L1+SSIM+LPIPS 监督，加 KL 正则。
输入：vertices/intersected（稀疏张量）与 mesh（GT 网格列表）；输出：terms（分项损失）与渲染快照。
契约：渲染监督要求同一组随机相机同时渲染 GT 与重建，否则对比无意义；lambda_* 为各项权重。
"""

from typing import *
import os
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils3d
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...renderers import MeshRenderer
from ...representations import Mesh
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, ssim, lpips


class ShapeVaeTrainer(BasicTrainer):
    """
    Trainer for Shape VAE
    
    形状 VAE 训练器：体素隐变量重建 + 随机相机可微渲染监督（mask/depth/normal）+ KL。
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        lambda_subdiv (float): Subdivision loss weight.
        lambda_intersected (float): Intersected loss weight.
        lambda_vertice (float): Vertice loss weight.
        lambda_kl (float): KL loss weight.
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
    """
    
    def __init__(
        self,
        *args,
        lambda_subdiv: float = 0.1,
        lambda_intersected: float = 0.1,
        lambda_vertice: float = 1e-2,
        lambda_mask: float = 1,
        lambda_depth: float = 10,
        lambda_normal: float = 1,
        lambda_kl: float = 1e-6,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        render_resolution: float = 1024,
        camera_randomization_config: dict = {
            'radius_range': [2, 100],
        },
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_subdiv = lambda_subdiv
        self.lambda_intersected = lambda_intersected
        self.lambda_mask = lambda_mask
        self.lambda_vertice = lambda_vertice
        self.lambda_depth = lambda_depth
        self.lambda_normal = lambda_normal
        self.lambda_kl = lambda_kl
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.camera_randomization_config = camera_randomization_config
        
        self.renderer = MeshRenderer({'near': 1, 'far': 3, 'resolution': render_resolution}, device=self.device)
        
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        均衡可恢复采样 + 多进程 DataLoader；同样在 collate 阶段按 batch_split 切分。
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
        
    def _randomize_camera(self, num_samples: int):
        # 随机相机：半径以 1/r^2 均匀采样（近似按面积均匀），fov = 2·asin(0.5/r) 保证物体占满视野；
        # 相机朝向原点、up 轴取 +z；near 在 [r-1, r] 内随机以拉开深度范围
        # sample radius and fov
        r_min, r_max = self.camera_randomization_config['radius_range']
        k_min = 1 / r_max**2
        k_max = 1 / r_min**2
        ks = torch.rand(num_samples, device=self.device) * (k_max - k_min) + k_min
        radius = 1 / torch.sqrt(ks)
        fov = 2 * torch.arcsin(0.5 / radius)
        origin = radius.unsqueeze(-1) * F.normalize(torch.randn(num_samples, 3, device=self.device), dim=-1)
        
        # build camera
        extrinsics = utils3d.torch.extrinsics_look_at(origin, torch.zeros_like(origin), torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device))
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        near = [np.random.uniform(r - 1, r) for r in radius.tolist()]
        
        return {
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'near': near,
        }
        
    def _render_batch(self, reps: List[Mesh], extrinsics: torch.Tensor, intrinsics: torch.Tensor, near: List,
                      return_types=['mask', 'normal', 'depth']) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
        Returns: 
            a dict with
                mask : [N x 1 x H x W] tensor of rendered masks
                normal : [N x 3 x H x W] tensor of rendered normals
                depth : [N x 1 x H x W] tensor of rendered depths
        """
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            # 每个样本单独设 near/far（far = near + 2，与 MeshRenderer 的配置对应）
            self.renderer.rendering_options['near'] = near[i]
            self.renderer.rendering_options['far'] = near[i] + 2
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types)
            # mask/depth 渲染结果缺通道维，这里补一维便于后续 stack/比较
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    def training_losses(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        mesh: List[Mesh],
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            vertices (SparseTensor): vertices of each active voxel
            intersected (SparseTensor): intersected flag of each active voxel
            mesh (List[Mesh]): the list of meshes to render

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # 编码：重参数化采样并返回 mean/logvar；解码同时给出重建网格与细分预测
        z, mean, logvar = self.training_models['encoder'](vertices, intersected, sample_posterior=True, return_raw=True)
        recon, pred_vertice, pred_intersected, subs_gt, subs = self.training_models['decoder'](z, intersected)
        
        terms = edict(loss = 0.0)
        
        # direct regression
        # 直接回归项：相交标记用 BCE（先 flatten），顶点坐标用 MSE
        if self.lambda_intersected > 0:
            terms["direct/intersected"] = F.binary_cross_entropy_with_logits(pred_intersected.feats.flatten(), intersected.feats.flatten().float())
            terms["loss"] = terms["loss"] + self.lambda_intersected * terms["direct/intersected"]
        if self.lambda_vertice > 0:
            terms["direct/vertice"] = F.mse_loss(pred_vertice.feats, vertices.feats)
            terms["loss"] = terms["loss"] + self.lambda_vertice * terms["direct/vertice"]
            
        # subdivision prediction loss
        # 逐级细分预测的 BCE（subs_gt 为各层真值，subs 为各层预测）
        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f"bce_sub{i}"] = F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            terms["loss"] = terms["loss"] + self.lambda_subdiv * terms[f"bce_sub{i}"]
            
        # rendering loss
        # 同一组随机相机分别渲染 GT 与重建网格（mask/depth/normal），保证逐像素可比
        cameras = self._randomize_camera(len(mesh))
        gt_renders = self._render_batch(mesh, **cameras, return_types=['mask', 'normal', 'depth'])
        pred_renders = self._render_batch(recon, **cameras, return_types=['mask', 'normal', 'depth'])
        # mask/depth 用 L1；normal 用 L1 + SSIM + LPIPS 组合，权重由 lambda_* 控制
        terms['render/mask'] = l1_loss(pred_renders['mask'], gt_renders['mask'])
        terms['render/depth'] = l1_loss(pred_renders['depth'], gt_renders['depth'])
        terms['render/normal/l1'] = l1_loss(pred_renders['normal'], gt_renders['normal'])
        terms['render/normal/ssim'] = 1 - ssim(pred_renders['normal'], gt_renders['normal'])
        terms['render/normal/lpips'] = lpips(pred_renders['normal'], gt_renders['normal'])
        terms['loss'] = terms['loss'] + \
                        self.lambda_mask * terms['render/mask'] + \
                        self.lambda_depth * terms['render/depth'] + \
                        self.lambda_normal * (terms['render/normal/l1'] + self.lambda_ssim * terms['render/normal/ssim'] + self.lambda_lpips * terms['render/normal/lpips'])
       
        # KL regularization     
        # KL 正则：约束后验接近标准正态，权重 lambda_kl
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
            
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        data = next(iter(dataloader))
        # Adjust num_samples to actual batch size (avoid empty SparseTensor slice)
        if 'mesh' in data:
            num_samples = min(num_samples, len(data['mesh']))
        elif 'vertices' in data:
            num_samples = min(num_samples, data['vertices'].shape[0])

        # inference
        # 两条解码路径都做快照：y 用训练态 decoder（带 intersected 输入），
        # y2 用 eval 态 decoder（不带 intersected，即推理时的调用方式），便于对比
        gts = []
        recons = []
        recons2 = []
        self.models['encoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            args = {k: v[i:i+batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            z = self.models['encoder'](args['vertices'], args['intersected'])
            self.models['decoder'].train()
            y = self.models['decoder'](z, args['intersected'])[0]
            z.clear_spatial_cache()
            self.models['decoder'].eval()
            y2 = self.models['decoder'](z)
            gts.extend(args['mesh'])
            recons.extend(y)
            recons2.extend(y2)
        self.models['encoder'].train()
        self.models['decoder'].train()
        
        # 渲染 GT / 两种重建的法线图做可视化；相机在 eval 结束后统一随机
        cameras = self._randomize_camera(num_samples)
        gt_renders = self._render_batch(gts, **cameras, return_types=['normal'])
        recons_renders = self._render_batch(recons, **cameras, return_types=['normal'])
        recons2_renders = self._render_batch(recons2, **cameras, return_types=['normal'])
        
        sample_dict = {
            'gt': {'value': gt_renders['normal'], 'type': 'image'},
            'rec': {'value': recons_renders['normal'], 'type': 'image'},
            'rec2': {'value': recons2_renders['normal'], 'type': 'image'},
        }
            
        return sample_dict
