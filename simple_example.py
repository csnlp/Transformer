import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import clones, attention, subsequent_mask
from model import *
"""
This task if for src-tgt copy task
"""
def data_gen(V, batch_size, nbatches):
    # data_batch generator
    # V can be see as the vocabulary size
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        # The start element of each samples is initialized as 1, similar as 
        # <SOS> of sentence in NLP
        data[:, 0] = 1
        src = Variable(data, requires_grad = False)
        tgt = Variable(data, requires_grad = False)
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        x: output of decoder, x.size(): (batch_size, max_len, vocab)
        x: y.size(): (batch_size, max_len, vocab)
        norm: 
        """
        x = self.generator(x)
        x = x.contiguous().view(-1, x.size(-1)) # size(x): (batch_size * max_len, vocab)
        y = y.contiguous().view(-1)   #
        loss = self.criterion(x, y) / norm

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm

# V is vocabulary size
V = 100
batch_size = 30
nbatches = 20
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
d_model = model.src_embed[0].d_model
model_opt = NoamOpt(d_model, 1, 400, opt)
loss_compute_train = SimpleLossCompute(model.generator, criterion, model_opt)
loss_compute_eval = SimpleLossCompute(model.generator, criterion, None)

for epoch in range(10):
    model.train()
    data_iter = data_gen(V, batch_size, nbatches)
    #print("model train: the generator size is {}".format(sum(1 for x in data_iter)))
    run_epoch(data_iter, model, loss_compute_train)
    model.eval()
    
    data_iter = data_gen(V, batch_size, nbatches)
    #print("model eval: the generator size is {}".format(sum(1 for x in data_iter)))
    print(run_epoch(data_iter, model, loss_compute_eval))

    

