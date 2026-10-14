#!/usr/bin/env python3
"""
应用 TRELLIS.2 微调所需的代码补丁
===================================
1. DINOv3 model.layer → model.model.layer (transformers >= 5.14 兼容)
2. SparseTensor 空切片修复 (Flow run_snapshot num_samples 裁剪)
3. Shape VAE run_snapshot DataLoader 修复 (iter/dataparallel bug)

用法:
    python scripts/apply_patches.py
    python scripts/apply_patches.py --reverse   # 撤销补丁
"""
import os
import sys
import argparse


def patch_dinov3_layer(reverse=False):
    """修复 DINOv3 在 transformers 5.14+ 中 layer 路径变化"""
    files_to_patch = [
        'trellis2/trainers/flow_matching/mixins/image_conditioned.py',
        'trellis2/modules/image_feature_extractor.py',
    ]

    for fpath in files_to_patch:
        if not os.path.exists(fpath):
            print(f'  [跳过] 文件不存在: {fpath}')
            continue

        with open(fpath, 'r') as f:
            content = f.read()

        if reverse:
            new_content = content.replace(
                'enumerate(self.model.model.layer):',
                'enumerate(self.model.layer):'
            )
        else:
            new_content = content.replace(
                'enumerate(self.model.layer):',
                'enumerate(self.model.model.layer):'
            )

        if new_content != content:
            with open(fpath, 'w') as f:
                f.write(new_content)
            action = '已撤销' if reverse else '已修补'
            print(f'  ✓ {action}: {fpath}')
        else:
            status = '已修补' if 'self.model.model.layer' in content else '无需修补'
            print(f'  - {status}: {fpath}')


def patch_sparse_slice(reverse=False):
    """修复 SparseTensor.__getitem__ 空切片导致 torch.cat([]) 崩溃"""
    fpath = 'trellis2/trainers/flow_matching/sparse_flow_matching.py'

    if not os.path.exists(fpath):
        print(f'  [跳过] 文件不存在: {fpath}')
        return

    with open(fpath, 'r') as f:
        content = f.read()

    # 要匹配和替换的代码片段
    old_snippet = (
        'data = next(iter(dataloader))\n'
        '\n'
        '        # inference\n'
        '        sampler = self.get_sampler()'
    )

    new_snippet = (
        'data = next(iter(dataloader))\n'
        '\n'
        '        # Adjust num_samples to actual batch size (avoid empty SparseTensor slice)\n'
        '        num_samples = min(num_samples, data[\'x_0\'].shape[0])\n'
        '\n'
        '        # inference\n'
        '        sampler = self.get_sampler()'
    )

    if reverse:
        source, target = new_snippet, old_snippet
    else:
        source, target = old_snippet, new_snippet

    if source in content:
        new_content = content.replace(source, target)
        with open(fpath, 'w') as f:
            f.write(new_content)
        action = '已撤销' if reverse else '已修补'
        print(f'  ✓ {action}: {fpath} (SparseTensor 空切片)')
    else:
        if 'num_samples = min(num_samples, data[' in content:
            status = '已修补' if not reverse else '?'
            print(f'  - {status}: {fpath}')
        else:
            print(f'  [警告] 未找到匹配代码段: {fpath}')
            print(f'         可能需要手动修改')


def patch_shape_vae_snapshot(reverse=False):
    """修复 Shape VAE run_snapshot 的 DataLoader 迭代 bug (与 Flow 同模式)"""
    fpath = 'trellis2/trainers/vae/shape_vae.py'

    if not os.path.exists(fpath):
        print(f'  [跳过] 文件不存在: {fpath}')
        return

    with open(fpath, 'r') as f:
        content = f.read()

    old_snippet = (
        '        dataloader = DataLoader(\n'
        '            copy.deepcopy(self.dataset),\n'
        '            batch_size=batch_size,\n'
        '            shuffle=True,\n'
        '            num_workers=1,\n'
        '            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, \'collate_fn\') else None,\n'
        '        )\n'
        '\n'
        '        # inference\n'
        '        gts = []\n'
        '        recons = []\n'
        '        recons2 = []\n'
        '        self.models[\'encoder\'].eval()\n'
        '        for i in range(0, num_samples, batch_size):\n'
        '            batch = min(batch_size, num_samples - i)\n'
        '            data = next(iter(dataloader))\n'
        '            args = {k: v[:batch] for k, v in data.items()}'
    )

    new_snippet = (
        '        dataloader = DataLoader(\n'
        '            copy.deepcopy(self.dataset),\n'
        '            batch_size=num_samples,\n'
        '            shuffle=True,\n'
        '            num_workers=0,\n'
        '            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, \'collate_fn\') else None,\n'
        '        )\n'
        '\n'
        '        data = next(iter(dataloader))\n'
        '        # Adjust num_samples to actual batch size (avoid empty SparseTensor slice)\n'
        '        if \'mesh\' in data:\n'
        '            num_samples = min(num_samples, len(data[\'mesh\']))\n'
        '        elif \'vertices\' in data:\n'
        '            num_samples = min(num_samples, data[\'vertices\'].shape[0])\n'
        '\n'
        '        # inference\n'
        '        gts = []\n'
        '        recons = []\n'
        '        recons2 = []\n'
        '        self.models[\'encoder\'].eval()\n'
        '        for i in range(0, num_samples, batch_size):\n'
        '            batch = min(batch_size, num_samples - i)\n'
        '            args = {k: v[i:i+batch] for k, v in data.items()}'
    )

    if reverse:
        source, target = new_snippet, old_snippet
    else:
        source, target = old_snippet, new_snippet

    if source in content:
        new_content = content.replace(source, target, 1)
        with open(fpath, 'w') as f:
            f.write(new_content)
        action = '已撤销' if reverse else '已修补'
        print(f'  ✓ {action}: {fpath} (Shape VAE snapshot)')
    else:
        if 'batch_size=num_samples' in content and 'num_workers=0' in content:
            status = '已修补' if not reverse else '?'
            print(f'  - {status}: {fpath}')
        else:
            print(f'  [警告] 未找到匹配代码段: {fpath}')
            print(f'         可能需要手动修改')


def clean_pycache():
    """清理所有 __pycache__ 目录"""
    import shutil
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(cache_path, ignore_errors=True)
            count += 1
    if count > 0:
        print(f'  ✓ 已清理 {count} 个 __pycache__ 目录')
    else:
        print(f'  - 无 __pycache__ 需要清理')


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 微调代码补丁')
    parser.add_argument('--reverse', action='store_true', help='撤销所有补丁')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)

    action = '撤销补丁' if args.reverse else '应用补丁'
    print(f'{"=" * 60}')
    print(f'  {action}')
    print(f'{"=" * 60}')

    clean_pycache()
    patch_dinov3_layer(reverse=args.reverse)
    patch_sparse_slice(reverse=args.reverse)
    patch_shape_vae_snapshot(reverse=args.reverse)

    print()
    print('  完成!')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
