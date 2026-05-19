import triton
import triton.language as tl
import torch
import torch.nn.functional as F

def softmax(X, mask=None, mask_type=None, causal=False, log=False):
    if X.dtype == torch.float16:
        IS_FP16 = True
    else:
        IS_FP16 = False

    if mask is not None:
        mask = mask.to(X.device)

    X_shape = X.shape
    grid = (X_shape[0] * X_shape[1], 1, 1)
    block_size = 1024
    depth = triton.next_power_of_2(X_shape[2])

    Y = torch.empty_like(X)
    triton._C._softmax_kernel(X, Y, X.stride(0), X.stride(1), X.stride(2), Y.stride(0), Y.stride(1), Y.stride(2),
                              mask, mask.stride(0), mask.stride(1), mask.stride(2), mask_type, causal, log, depth,
                              block_size, IS_FP16)

    return Y
