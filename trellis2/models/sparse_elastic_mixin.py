"""稀疏 Transformer 的显存弹性（activation checkpointing）混入类。

把 ElasticModuleMixin 的 mem_ratio 语义适配到稀疏张量：稀疏 batch 的“输入规模”
用有效体素数 x.feats.shape[0] 衡量，据此决定给多少个 block 打开 checkpoint 以省显存。
"""
from contextlib import contextmanager
from typing import *
import math
from ..modules import sparse as sp
from ..utils.elastic_utils import ElasticModuleMixin


class SparseTransformerElasticMixin(ElasticModuleMixin):
    def _get_input_size(self, x: sp.SparseTensor, *args, **kwargs):
        # 稀疏张量没有固定 batch 维度，用非零体素个数代表规模
        return x.feats.shape[0]
    
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        # mem_ratio 越小，开启 checkpoint 的 block 越多（1.0 表示不省显存）
        if mem_ratio == 1.0:
            yield 1.0
            return
        num_blocks = len(self.blocks)
        # 前 num_checkpoint_blocks 个 block 打开重计算，返回实际能达到的 mem_ratio
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = False
