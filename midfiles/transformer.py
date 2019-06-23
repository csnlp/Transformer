import math, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src_embed is the 
        src_embedding = self.src_embed(src)
        return self.encoder(src_embedding, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embedding = self.tgt_embed(tgt)
        return self.decoder(tgt_embedding, memory, src_mask, tgt_mask)

