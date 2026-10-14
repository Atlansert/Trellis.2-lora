import csv, json, os
from typing import *
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageConditionedSparseStructureLatentViews(Dataset):
    """TRELLIS.1-style paired supervision: each rendered view is paired with
    the SS latent of the mesh rotated into that view's camera frame."""
    def __init__(self, roots: str, *, image_size=512, min_aesthetic_score=0.0,
                 crop_scale=1.2, skip_snapshot=True,
                 num_views=24, latent_frame='view',
                 canonical_latent_root=None, min_view_aspect=0.0,
                 max_view_offaxis=None, **kwargs):
        cfg=json.loads(roots) if isinstance(roots,str) else roots
        self.image_size=image_size; self.num_views=num_views
        self.latent_frame=str(latent_frame)
        if self.latent_frame not in ('view', 'canonical'):
            raise ValueError("latent_frame must be 'view' or 'canonical'")
        self.canonical_latent_root=canonical_latent_root
        self.crop_scale=float(crop_scale)
        self.min_view_aspect=float(min_view_aspect)
        if self.min_view_aspect < 0:
            raise ValueError('min_view_aspect must be >= 0 (0 = 不过滤视角)')
        # 目标长轴相对正侧视轴的偏角（度）。None 表示不按姿态过滤。
        # 宽高比过滤会把 30°/45° 斜侧视留下来，那些图看起来都像长车，
        # 目标朝向却差几十度，模型会把车身平均成圆管。
        self.max_view_offaxis = None if max_view_offaxis is None else float(max_view_offaxis)
        if self.max_view_offaxis is not None and self.max_view_offaxis < 0:
            raise ValueError('max_view_offaxis must be >= 0')
        if self.crop_scale < 1.0:
            raise ValueError('crop_scale must be >= 1.0 (1.0 = 物体占满画面，与推理一致)')
        if min_aesthetic_score < 0:
            raise ValueError('min_aesthetic_score must be non-negative')
        self.value_range=(0,1); self.normalization=None
        # 默认关闭快照：SS 是 8 通道 3D latent，通用可视化不适用；且 trainer
        # 的 snapshot() 调用未包 try，失败会终止训练。配好 snapshot_sampler_params
        # 后可显式传 skip_snapshot=false，拿到与线上同口径的中间图。
        self.skip_snapshot = bool(skip_snapshot)
        # TRELLIS.1 stores one instance per object and samples one view at
        # random in ``get_instance``.  Keep the same sampling semantics here
        # instead of expanding every object into 24 independent instances.
        self.instances=[]
        self._view_aspect={}
        self._view_offaxis={}
        for _, r in cfg.items():
            base=r.get('base', next(iter(r.values())) if r else '')
            latent_root=r.get('ss_view_latent')
            render_root=r.get('render_cond', os.path.join(base,'renders_cond'))
            if not latent_root: raise ValueError('data_dir requires ss_view_latent')
            if self.min_view_aspect > 0:
                cap=os.path.join(render_root, '_view_aspect.json')
                if os.path.isfile(cap):
                    self._view_aspect.update(json.load(open(cap)))
            if self.max_view_offaxis is not None:
                cap=os.path.join(render_root, '_view_offaxis.json')
                if not os.path.isfile(cap):
                    raise FileNotFoundError(
                        f'max_view_offaxis={self.max_view_offaxis} 需要 {cap}')
                self._view_offaxis.update(json.load(open(cap)))
            eligible=None
            metadata_path=os.path.join(base, 'metadata.csv')
            if os.path.isfile(metadata_path):
                eligible=set()
                with open(metadata_path, newline='') as mf:
                    for row in csv.DictReader(mf):
                        try:
                            score=float(row.get('aesthetic_score', 0))
                        except (TypeError, ValueError):
                            score=0.0
                        rendered=str(row.get('cond_rendered', '')).lower() == 'true'
                        if rendered and score >= min_aesthetic_score:
                            eligible.add(row.get('sha256', ''))
            for obj in sorted(os.listdir(latent_root)):
                if eligible is not None and obj not in eligible: continue
                od=os.path.join(latent_root,obj)
                if not os.path.isdir(od): continue
                with open(os.path.join(render_root,obj,'transforms.json')) as f: frames=json.load(f)['frames']
                n=min(self.num_views,len(frames))
                if self.latent_frame=='canonical':
                    # 规范帧：目标是每个对象唯一的 latent，条件用任意一张渲染视图
                    cp=os.path.join(self.canonical_latent_root, f'{obj}.npz')
                    if not os.path.exists(cp):
                        continue
                    valid=[i for i in range(n)
                           if os.path.exists(os.path.join(render_root,obj,frames[i]['file_path']))]
                    # 规范帧目标是「最侧视」那一帧。端视/大斜视的长高比和这个目标相反，
                    # 留着会让模型忽略输入图的比例、把截面越训越瘦。只保留侧视。
                    if self.min_view_aspect > 0:
                        table=self._view_aspect.get(obj)
                        if table is None:
                            valid=[i for i in valid
                                   if self._foreground_aspect(os.path.join(
                                       render_root, obj, frames[i]['file_path']))
                                   >= self.min_view_aspect]
                        else:
                            valid=[i for i in valid
                                   if table.get(os.path.basename(frames[i]['file_path']), 0)
                                   >= self.min_view_aspect]
                    valid=self._filter_offaxis(obj, frames, valid)
                    if valid:
                        self.instances.append((render_root,obj,cp,valid))
                else:
                    # 逐视角相机帧：每张图配自己的 latent。端视的长轴不在 index1，
                    # 和 TRELLIS.2 底座的规范帧先验冲突，所以同样只保留侧视。
                    valid=[i for i in range(n)
                           if os.path.exists(os.path.join(od, f'{i:03d}.npz'))
                           and os.path.exists(os.path.join(render_root, obj, frames[i]['file_path']))]
                    if self.min_view_aspect > 0:
                        table=self._view_aspect.get(obj)
                        if table is None:
                            valid=[i for i in valid
                                   if self._foreground_aspect(os.path.join(
                                       render_root, obj, frames[i]['file_path']))
                                   >= self.min_view_aspect]
                        else:
                            valid=[i for i in valid
                                   if table.get(os.path.basename(frames[i]['file_path']), 0)
                                   >= self.min_view_aspect]
                    valid=self._filter_offaxis(obj, frames, valid)
                    if valid:
                        self.instances.append((render_root,obj,od,valid))
        print(f'{self.__class__.__name__}: {len(self.instances)} objects, '
              f'{sum(len(x[3]) for x in self.instances)} view pairs, '
              f'latent_frame={self.latent_frame}, '
              f'min_view_aspect={self.min_view_aspect}, '
              f'max_view_offaxis={self.max_view_offaxis}', flush=True)
    def _filter_offaxis(self, obj, frames, valid):
        if self.max_view_offaxis is None:
            return valid
        table=self._view_offaxis.get(obj)
        if not table:
            return []
        kept=[]
        for i in valid:
            ang=table.get(os.path.basename(frames[i]['file_path']))
            if ang is not None and float(ang) <= self.max_view_offaxis:
                kept.append(i)
        return kept
    @staticmethod
    def _foreground_aspect(path):
        im=Image.open(path).convert('RGBA').resize((64, 64), Image.Resampling.BOX)
        a=np.array(im.getchannel('A'))
        ys,xs=np.nonzero(a>32)
        if len(xs)<4:
            return 0.0
        return (xs.max()-xs.min()+1)/max(1, ys.max()-ys.min()+1)
    def __len__(self): return len(self.instances)
    def _image(self,p):
        im=Image.open(p).convert('RGBA'); a=np.array(im.getchannel(3)); ys,xs=np.nonzero(a)
        if len(xs):
            cx=(xs.min()+xs.max())/2; cy=(ys.min()+ys.max())/2
            hs=max(xs.max()-xs.min(),ys.max()-ys.min())/2*self.crop_scale
            im=im.crop((int(cx-hs),int(cy-hs),int(cx+hs),int(cy+hs)))
        im=im.resize((self.image_size,self.image_size),Image.Resampling.LANCZOS)
        a=torch.from_numpy(np.array(im.getchannel(3))).float()/255
        x=torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()/255
        return x*a.unsqueeze(0)
    def __getitem__(self,i):
        rr,obj,od,valid=self.instances[i]
        v=int(np.random.choice(valid))
        with open(os.path.join(rr,obj,'transforms.json')) as f: fr=json.load(f)['frames'][v]
        if self.latent_frame=='canonical':
            lp=od                      # od 已是 <obj>.npz
        else:
            lp=os.path.join(od,f'{v:03d}.npz')
        latent=np.load(lp)
        key='mean' if 'mean' in latent.files else 'z'
        return {'x_0':torch.from_numpy(latent[key]).float(),
                'cond':self._image(os.path.join(rr,obj,fr['file_path']))}

    def visualize_sample(self, x_0):
        """Return a valid image grid for generic trainer snapshots.

        Per-view SS latents are 8-channel 3-D tensors; they are not directly
        displayable by torchvision.  A central occupancy proxy keeps snapshot
        bookkeeping from interfering with the actual training path.
        """
        x = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        return x[:, :3, :, :, x.shape[-1] // 2]
