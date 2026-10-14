"""文本条件 mixin：用 CLIP 文本编码器把文本 prompt 变成条件张量。

职责：惰性加载 CLIPTextModel + Tokenizer，提供 encode_text，并把编码结果接到
CFG/flow matching 的条件接口上（负条件用空串的编码）。
输入：List[str]（batch 的文本 prompt）；输出：last_hidden_state 文本嵌入（GPU Tensor）。
契约：仅支持「字符串列表」形式的条件；tokenizer 固定 max_length=77 截断/补齐，
负条件 null_cond 在模型初始化时用空串预先算好。
"""

from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import AutoTokenizer, CLIPTextModel

from ....utils import dist_utils


class TextConditionedMixin:
    """
    Mixin for text-conditioned models.
    
    文本条件 mixin：负责 CLIP 文本编码与负条件（空串）构造。

    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(self, *args, text_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None     # the model is init lazily
        
    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        惰性初始化：只在首次需要时加载 CLIP 文本模型与分词器，并预计算空串负条件。
        """
        # load model
        # 只在 local master 上先加载，避免多卡同时下载/读取权重；随后各 rank 各自把模型搬到 GPU
        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])   # 空串嵌入 = CFG 的负条件
        
    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        把文本列表编码为 CLIP 最后一层隐状态（batch, 77, dim），全程 no_grad。
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        训练用：编码文本并把空串嵌入重复到同一 batch 作为 neg_cond，再交给上层 CFG mixin。
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        推理用：同上，但不做条件丢弃。
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
