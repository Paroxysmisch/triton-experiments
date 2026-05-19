import torch
import triton

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    n, in_features = input.shape
    out_features = weight.shape[0]
    
    if normalized_shape is None:
        normalized_shape = in_features
    
    assert isinstance(normalized_shape, int), "normalized_shape must be an integer"
    
    x = input.contiguous()
    w = weight.contiguous()
    b = bias.contiguous() if bias is not None else None
    gamma = weight.contiguous() if elementwise_affine else None
    beta = bias.contiguous() if elementwise_affine else None
    
    out = torch.empty((n, out_features), device=input.device, dtype=input.dtype)
    
    n, m, k = n, out_features, in_features
    norm_n = normalized_shape
    
    fused_layer_norm_relu_linear(x, w, b, gamma, beta, out, n, m, k, norm_n, eps)
    
    return out
