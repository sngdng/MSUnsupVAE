import copy
import sys
from pathlib import Path

import torch
from transformers import T5Tokenizer

from d2t.data.formatting import STYLE_TOKEN, DataFormat


class VAET5Tokenizer(T5Tokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, use_style_token=True, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_kg_vocabulary()
        inst.init_dart_vocabulary()
        inst.init_totto_vocabulary()
        if use_style_token:
            inst.add_tokens(STYLE_TOKEN, special_tokens=True)
        return inst

    def init_kg_vocabulary(self):
        kg_tokens = [
            DataFormat.HEAD_TOKEN,
            DataFormat.TYPE_TOKEN,
            DataFormat.TAIL_TOKEN,
            DataFormat.BLANK_TOKEN,
        ]
        self.add_tokens(kg_tokens)

    def init_dart_vocabulary(self):
        self.add_tokens(DataFormat.DART_TOKENS)

    def init_totto_vocabulary(self):
        self.add_tokens(DataFormat.TOTTO_TOKENS)
