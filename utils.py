import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
"""
INPUT: 
    query.size(): (batch_size, h, max_len, d_k)
    key.size(): (batch_size, h, max_len, d_k)
    value.size(): (batch_size, h, max_len, d_v)
OUTPUT:
    torch.matmul(p_attn, value).size(): (batch_size, h, max_len, d_v)
    p_attn.size(): (batch_size, h, max_len, max_len, max_len)
"""
def attention(query, key, value, mask=None, dropout=None):
    #print("The size of query, key, value is {}, {}, {}".format(query.size(), key.size(), value.size()))
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

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

