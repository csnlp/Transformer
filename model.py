# -*- coding:utf-8 -*-
import math
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import clones, attention, subsequent_mask

deepcopy = copy.deepcopy

class Generator(nn.Module):
    # Generator = Linear Layer(Projection) + Softmax Layer(for output the probability distribution of each words in vocubulary)

    def __init__(self, dimen_model, dimen_vocab):
        # dimen_model: the dimension of decoder output
        # dimen_vocab: the dimension of vocabulary 
        super(Generator, self).__init__()
        self.projection_layer = nn.Linear(dimen_model, dimen_vocab)

    def forward(self, x):
        # input.size(): (batch_size, max_len, d_model)
        # output.size(): (batch_size, max_len, d_vocab)
        return F.log_softmax(self.projection_layer(x), dim=-1)

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

"""
Encoder is composed of several identical layers, each layer is composed of two sublayers
1. Multi-Head Self-attentioin : {input: x, output: LayerNorm(x + Sublayer(x))}
2. Position-wise Fully-connected feed-forward network: {input: x, output: LayerNorm(x + Sublayer(x))}
Each sublayer is employed with a residual connection, and then followed by layer normalization
"""
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
"""
LayerNorm is the implement code of Layer Normalization(LN) for each of two sublayers 
For more information about Layer Normalization and Normalization, please see my Blog
"""
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # a_2, b_2 is trainable to scale means and std variance
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

"""
input: x
SublayerConnection is the operation about(orderly): 
    1. Layernormalize the input
    2. Sublayer function
    3. Dropout: See Section 5.4 Regularization: Residual Dropout
                <We apply dropout to the output of each sub-layer, before it it is added to the 
                sublayer input normalized>
                
                Actually, The add action between the output of each sub-layer and the sub-layer 
                input is known as Residual Connection
    4. Residual Connection
"""
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    # sublayer is a function defined by self attention or feed forward 
    def forward(self, x, sublayer):
        # Normalization
        norm_x = self.norm(x)
        # Sublayer function
        sublayer_x = sublayer(norm_x)
        # Dropout function
        dropout_x = self.dropout(x)
        # Residual connection
        return x + dropout_x

"""
EncoderLayer is the single piece layer in Encoder
each EncoderLayer is composed of two sublayer
1. self_attention
2. feed forward
"""
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # self.self_attn is the Multi-Head Attention Layer
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayerconnections = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # self attention
        self_attention_x = self.sublayerconnections[0](x, lambda x: self.self_attn(x, x, x, mask))
        # feed forward
        feed_forward_x = self.sublayerconnections[1](self_attention_x, self.feed_forward)
        return feed_forward_x

"""
DecoderLayer is the single piece layer in Encoder
each EncoderLayer is composed of two sublayer
1. Self attention
2. Convention attention 
3. Feed Forward
"""
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayerconnections = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        self_attention_x = self.sublayerconnections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        src_attention_x = self.sublayerconnections[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        feed_forward_x = self.sublayerconnections[2](x, lambda x: self.feed_forward)
        return feed_forward_x


"""
h is the number of the parallel attention layers. In paper, h=8
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        #print d_model, h
        assert d_model % h == 0
        
        # self.d_k is the reduced dimension of each parallel attention
        self.d_k = d_model // h
        self.h = h

        # self.linears is a list consists of 4 projection layers
        # self.linears[0]: Concat(W^Q_i), where i \in [1,...,h]. 
        # self.linears[1]: Concat(W^K_i), where i \in [1,...,h]. 
        # self.linears[2]: Concat(W^K_i), where i \in [1,...,h]. 
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query.size() = key.size() = value.size() = (batch_size, max_len, d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        """
        do all the linear projection, after this operation
        query.size() = key.size() = value.size() = (batch_size, self.h, max_len, self.d_k)
        """
        query, key, value = \
                [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for 
                        linear, x in zip(self.linears, (query, key, value))]
        """
        x.size(): (batch_size, h, max_len, d_v)
        self.attn.size(): (batch_size, h, max_len, d_v)
        """
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        """
        x.transpose(1,2).size(): (batch_size, max_len, h, d_v)
        the transpose operation is necessary
        x.size: (batch_size, max_len, h*d_v)
        """
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # self.linears[-1] \in R^{hd_v \times d_{model}}
        return self.linears[-1](x)

"""
There are three types of Multi-Head Attention
1. Self-Attention in the Encoder
   Query: the output of the previous layer
   Key  : the output of the previous layer
   Value: the output of the previous layer

2. Self-Attention in the Decoder
   Query: the output of the previous layer
   Key  : the output of the previous layer
   Value: the output of the previous layer
   
3. Attention between Encoder and Decoder
   Query: the output of the previous decoder layer
   Key  : the output of the encoder
   Value: the output of the encoder
"""


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        w1_x = self.w_1(x)
        relu_x = F.relu(w1_x)
        dropout_x = self.dropout(relu_x)
        return self.w_2(dropout_x)


"""
Convet a one-hot vector to d_model vector
input  : batch_size * vocab
output : batch_size * d_model
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # the size of position is (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        """
        See Section 5.4 Regularization Residual Dropout: 
         <In addition, we apply dropout to the sums of the embeddings and the positional encoding
          in both the encoder and decoder stacks>
        """
        return self.dropout(x)


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

# LabelSmoothing is a regularization method 
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.trust_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.data.unsqueeze(-1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1:
            #print mask.squeeze()
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
        d_ff=2048, h=8, dropout=0.1):
    # Basic Components
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N)
    decoder = Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position))
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Batch(object):
    def __init__(self, src, trg, pad=0):
        # src.size(): (batch_size, max_len)
        self.src = src
        # src_mask.size(): (batch_size, 1, max_len)
        self.src_mask = (src != pad).unsqueeze(-2)

        # if target is not None
        if trg is not None: 
            # size(self.trg)   : (batch_size, max_len-1)
            self.trg = trg[:, :-1]
            # size(self.trg_y) : (batch_size, max_len-1)
            self.trg_y = trg[:, 1:]
            
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum().float()

    @staticmethod
    def make_std_mask(tgt, pad):
        # pad_tgt_mask.size(): (batch_size, 1, max_len-1)
        # pad_tgt_mask is for [padding] mask in each sentence
        pad_tgt_mask = (tgt != pad).unsqueeze(-2)
        length = tgt.size(-1)
        # sub_tgt_mask.size(): (max_len-1, max_len-1)
        # sub_tgt_mask is for [future word] mask
        sub_tgt_mask = subsequent_mask(length).type_as(pad_tgt_mask.data)
        
        # total_tgt_mask.size(): (batch_size, max_len-1, max_len-1)
        # total_tgt_mask is for padding and future words mask
        total_tgt_mask = pad_tgt_mask & sub_tgt_mask 
        return total_tgt_mask

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        #print('*' * 20)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        # tokens is used to sum all the generated tokens every 50 batches
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d  Loss: %f   Tokens per Sec: %f" 
                    %(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
        
    return total_loss / total_tokens

class NoamOpt:
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
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        tmp_a = step**(-0.5)
        tmp_b = step * self.warmup**(-1.5)
        return self.factor * (self.model_size)**(-0.5) * min(tmp_a, tmp_b)



        

