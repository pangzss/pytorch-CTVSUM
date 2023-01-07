import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules import *

class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_inp=1024, n_layers=4, n_head=1, d_k=64, d_v=64,
            d_model=256, d_inner=512, dropout=0., num_patches=300):

        super().__init__()
        self.n_layers = n_layers

        self.proj = nn.Linear(d_inp, d_model) 
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.unq_est = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
        )
    def forward(self, src_seq):
        # -- Forward
        enc_output = self.proj(src_seq)
        enc_output = self.layer_norm(enc_output)
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, _ = enc_layer(enc_output)
        scores = self.score(self.unq_est(enc_output.detach())[0])
        return enc_output, scores.squeeze(-1)

if __name__ == '__main__':
    model = TransformerEncoder()
    inp = torch.rand(1,300,1024)
    enc_output = model(inp)
    print(enc_output.shape)
