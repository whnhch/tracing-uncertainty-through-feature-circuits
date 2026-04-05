import torch
import torch.nn as nn
import torch.nn.functional as F

class Transcoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size

        # Encoder: maps MLP input to sparse features
        self.W_enc = nn.Linear(input_dim, dict_size, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        # Decoder: maps sparse features to MLP output
        self.W_dec = nn.Linear(dict_size, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # Tie decoder weights to unit norm (standard practice in dictionary learning I guess)
        # Reference:
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)

    def encode(self, x_in):
        # Shift by decoder bias before encoding
        shifted_x = x_in - self.b_dec
        # Calculate pre-activations and apply ReLU for sparsity
        pre_acts = self.W_enc(shifted_x) + self.b_enc
        return F.relu(pre_acts)

    def decode(self, f):
        return self.W_dec(f) + self.b_dec

    def forward(self, x_in):
        f = self.encode(x_in)
        x_out_pred = self.decode(f)
        return x_out_pred, f