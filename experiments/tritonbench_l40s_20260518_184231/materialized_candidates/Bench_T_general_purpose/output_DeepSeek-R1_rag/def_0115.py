import torch
import triton
import triton.language as tl
from typing import Optional

# Add tanh and its gradient computation to the existing activation imports
# Assuming these are added to k_activations or defined here
@triton.jit
def tanh_grad(x):
    tanh_x = tl.tanh(x)
    return 1.0 - tanh_x * tanh_x

@triton.jit
def kernel_fwd(
    C, ACT_INPUT, A, B, bias,
    M, N, K, CACHE_KEY_M, CACHE_KEY_N, CACHE_KEY_K,
    stride_cm, stride_am, stride_ak, stride_bn, stride_bk,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, A_ROWMAJOR: tl.constexpr, B_COLMAJOR: tl.constexpr,
    BIAS: tl.constexpr, SAVE_ACT_INPUT: tl.constexpr, ACTIVATION: tl.constexpr,
):
    # ... [Previous kernel_fwd code] ...
    
    # Add Tanh activation
    if ACTIVATION == "tanh":
        acc = tl.tanh(acc)
    
    # ... [Remaining kernel_fwd code] ...

@triton.jit
def kernel_bwd(
    C, ACT_INPUT, A, B,
    M, N, K, CACHE_KEY_M, CACHE_KEY_N, CACHE_KEY_K,
    stride_cm, stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, ACTIVATION: tl.constexpr,
):
    # ... [Previous kernel_bwd code] ...
    
    # Handle Tanh gradient
    if ACTIVATION == "tanh":
        act_input = tl.load(act_in_ptrs).to(acc.dtype)
        grad_tanh = tanh_grad(act_input)
        acc *= grad_tanh
    
    # ... [Remaining kernel_bwd code] ...

def triton_linear_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "id",
    save_act_input: bool = False,
) -> torch.Tensor:
    assert activation in ["id", "gelu", "gelu_approx", "squared_relu", "tanh"]  # Add "tanh"
    # ... [Rest of triton_linear_act code] ...

def triton_dgrad_act(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "id",
    act_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert activation in ["id", "gelu", "gelu_approx", "squared_relu", "tanh"]  # Add "tanh"
    # ... [Rest of triton_dgrad_act code] ...

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    output, act_input = triton_linear_act(
        input, weight, bias, activation='tanh', save_act_input=True
    )
    return output
