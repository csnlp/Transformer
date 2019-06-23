# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    # 实际上就是Attention is All You Need 中的Linear Layer + Softmax Layer
    # 也就是Linear Layer 其实是 Projection Layer, Softmax 在词典上概率
    # Generator = Linear Layer(Projection) + Softmax Layer(for output the probability distribution of each words in vocubulary)

    def __init__(self, dimen_model, dimen_vocab):
        # dimen_model: the dimension of decoder output
        # dimen_vocab: the dimension of vocabulary 
        super(Generator, self).__init__()
        self.projection_layer = nn.Linear(dimen_model, dimen_vocab)

    def forward(self, x):
        return F.log_softmax(self.projection_layer(x), dim=-1)
