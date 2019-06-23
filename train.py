import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *

deepcopy = copy.deepcopy

"""
The subsequent mask method is mainly for the decoder layer
"""
def subsequent_mask(size):
    attn_shape = (1, size, size)
    """
    np.triu: https://docs.scipy.org/doc/numpy/reference/generated/numpy.triu.html
    np.triu: upper triangle of an array
    for instance: if size = 4, and = 1 the subsequent_mask is 
    [[0, 1, 1, 1],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]
    ]
    """
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
        d_ff=2048, h=8, dropout=0.1):
    # Basic Components
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositioinalEncoding(d_model, dropout)
    
    encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N)
    decoder = Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position))
    generator = Generator(d_model, tgt_vocab)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Batch(object):
    def __init__(self, src, trg, pad=0):
        # src.size(): (n_batches, max_len)
        self.src = src
        # src_mask.size(): (n_batches, 1, max_len)
        self.src_mask = (src != pad).unsqueeze(-2)

        # if target is not None
        if trg is not None: 
            # size(self.trg)   : (n_batches, max_len-1)
            self.trg = trg[:, :-1]
            # size(self.trg_y) : (n_batches, max_len-1)
            self.trg_y = trg[:, 1:]
            
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # pad_tgt_mask.size(): (n_batches, 1, max_len-1)
        # pad_tgt_mask is for [padding] mask in each sentence
        pad_tgt_mask = (tgt != pad).unsqueeze(-2)
        length = tgt.size(-1)
        # sub_tgt_mask.size(): (max_len-1, max_len-1)
        # sub_tgt_mask is for [future word] mask
        sub_tgt_mask = subsequent_mask(length).type_as(pad_tgt_mask.data)
        
        # total_tgt_mask.size(): (n_batches, max_len-1, max_len-1)
        # total_tgt_mask is for padding and future words mask
        total_tgt_mask = pad_tgt_mask & sub_tgt_mask 
        return total_tgt_mask

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d  Loss: %f   Tokens per Sec: %f" 
                    %(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
        return total_loss / total_tokens

class Opt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        
