import torch
from torch.nn import functional as F

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', out=None):
    # Batch matrix multiplication
    Z1 = torch.bmm(input1, input2)

    # RMS normalization
    Z2 = F.normalize(Z1, p=2, dim=-1) * normalized_shape

    # GELU activation
    if approximate == 'tanh':
        Z3 = F.gelu(Z2, approximate='tanh')
    else:
        Z3 = F.gelu(Z2)

    # Dropout
    if training:
        Z = F.dropout(Z3, p=dropout_p, training=True)
    else:
        Z = Z3

    return Z
