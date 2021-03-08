import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import *
from enum import IntEnum
# class Dim(IntEnum):
#     batch = 0
#     seq = 1
#     feature =2

class MogLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, mog_iterations: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mog_iterations = mog_iterations
        self.WiH = Parameter(torch.Tensor(input_size, hidden_size*4))
        self.WhH = Parameter(torch.Tensor(hidden_size, hidden_size*4))
        self.BiH = Parameter(torch.Tensor(hidden_size*4))
        self.BhH = Parameter(torch.Tensor(hidden_size*4))

        self.Q = Parameter(torch.Tensor(hidden_size, input_size))
        self.R = Parameter(torch.Tensor(input_size, hidden_size))

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    # def mogrify(self, xt, ht):
    #     for i in range(1, self.mog_iterations+1):
    #         if (i % 2 == 0):
    #             ht = (2*torch.sigmoid(torch.matmul(xt, self.R))) * ht
    #         else:
    #             xt = (2*torch.sigmoid(torch.matmul(ht, self.Q))) * xt
    #     return xt, ht

    def mogrify(self, xt, ht):
        xt1 = xt
        ht1 = ht
        for i in range(1, self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(torch.matmul(xt1, self.R))) * ht
            else:
                xt = (2*torch.sigmoid(torch.matmul(ht1, self.Q))) * xt
        return xt, ht

    def forward(self, x, init_states):
        batch_size, seq_size = x.size()
        if init_states is None:
            HT = torch.zeros((batch_size, self.hidden_size)).to(x.device)
            CT = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        else:
            HT, CT = init_states

        # for t in range(seq_size):
        XT = x
        XT, HT = self.mogrify(XT, HT)
        gates = (XT @ self.WiH + self.BiH) + (HT @ self.WhH + self.BhH)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ft = torch.sigmoid(forgetgate)
        it = torch.sigmoid(ingate)
        Ct_candidate = torch.tanh(cellgate)
        ot = torch.sigmoid(outgate)

        CT = (ft * CT) + (it * Ct_candidate)
        HT = ot * torch.tanh(CT)
        return HT, CT