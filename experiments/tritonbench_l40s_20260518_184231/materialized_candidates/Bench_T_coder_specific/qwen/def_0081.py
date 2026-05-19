import triton
from typing import Union, Tuple
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 64}, num_stages=1, num_warps=4),
    ],
    key=['N', 'C', 'H_out', 'W_out']
)
def _adaptive_avg_pool2d_forward_kernel(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    H_out: int,
    W_out: int,
    stride: int,
    pad: int,
    BLOCK_SIZE_X: int = 32,
):
    adaptive_avg_pool2d_kernel[
        grid=(N * C,), block=(BLOCK_SIZE_X,), 
        x_ptr=x_ptr, y_ptr=y_ptr, N=N, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out, stride=stride, pad=pad
    ]

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    N, C, H_in, W_in = input.shape
    H_out, W_out = output_size
    
    y = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Apply adaptive average pooling
    _adaptive_avg_pool2d_forward_kernel(
        x_ptr=input.contiguous().view(-1).to(torch.float32),
        y_ptr=y.view(-1).to(torch.float32),
        N=N, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        stride=1, pad=0
    )
    
    # Apply sigmoid activation
    y_sigmoid = torch.sigmoid(y)
    
    return y_sigmoid
