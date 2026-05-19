import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function
from .triton_utils import get_kernel_meta, has_triton

@triton.jit
def _silu_and_mul(x, y):
    # Silu activation function
    x = tl.fdiv(x, (1 + tl.exp(-x)))
    # Element-wise multiplication
    return x * y

@triton.jit
def _silu(x):
    # Silu activation function
    return tl.fdiv(x, (1 + tl.exp(-x)))

@triton.jit
def _linear(x, weight, bias):
    # Linear layer computation
    return tl.dot(x, weight, out_dtype=x.dtype) + bias

@triton.jit
def _layer_norm(x, weight, bias, eps):
    # Layer normalization
    mean = tl.mean(x, axis=-1, keepdims=True)
    x = tl.where(x >= mean, x, mean)
    var = tl.sum((x - mean) * (x - mean), axis=-1, keepdims=True) / x.shape[-1]
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias

@triton.jit
def _dropout(x, p, seed, offset):
    # Dropout
    random = tl.rand(seed, offset)
    return tl.where(random > p, x / (1 - p), 0.0)

@triton.jit
def _bmm_rmsnorm_gelu_dropout_a(x, y, weight, ln_weight, ln_bias, dropout_p, seed, eps, training, approximate):
    # BMM, RMSNorm, GELU, and Dropout operation
    batch, seq, _ = x.shape
    out = torch.empty((batch, seq, y.shape[2]), dtype=x.dtype, device=x.device)
    offset = tl.arange(0, 256)
    trans_y = tl.trans(y)
    for i in range(0, batch):
        for j in range(0, seq, 256):
            p = j + offset
            mask = p < seq
            a = tl.load(x + i * seq + p, mask=mask)
            b = tl.load(trans_y + j * 256 + (p % 256), mask=mask)
            c = tl.dot(a, b, out_dtype=a.dtype)
            d = _layer_norm(c, ln_weight, ln_bias, eps)
            if approximate == "tanh":
                e = 0.5 * d * (1 + tl.tanh(_silu(d)))
            else:
                e = _silu_and_mul(d, d)
            if training:
                out_offset = i * seq * y.shape[2] + j * 256
                o = out.view(batch * seq, y.shape[2])
                o_ptr = o + out_offset + offset
                tl.store(o_ptr, e, mask=(offset < (seq * y.shape[2]) - out_offset))
            else:
                f = _dropout(e, dropout_p, seed, i)
                out_offset = i * seq * y.shape[2] + j * 256
                o = out.view(batch * seq, y.shape[2])
                o_ptr = o + out_offset + offset
                tl.store(o_ptr, f, mask=(offset < (seq * y.shape[2]) - out_offset))
    return out

def _bmm_rmsnorm_gelu_dropout(
    x: Tensor,
    y: Tensor,
    normalized_shape: int,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
    training: bool = True,
    approximate: str = "none",
    out: Tensor = None,
) -> Tensor:
    # Wrapper function for the Triton kernel
    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[1], y.shape[2]), dtype=x.dtype, device=x.device
        )
    if approximate not in ("none", "tanh"):
        raise ValueError(f"Invalid approximate value: {approximate}")
    if x.shape[0] != out.shape[0] or x.shape[1] != out.shape[1] or y.shape[2] != out.shape[2]:
        raise ValueError("Shape of tensors must match the operation")
    if dropout_p < 0 or dropout_p > 1:
        raise ValueError("Dropout probability must be between 0 and 1")
    if not isinstance(normalized_shape, int):
        raise ValueError("Only integer value is allowed for normalized_shape")
    if not has_triton():
        raise RuntimeError("Triton kernel requires triton to be installed")
    seed = torch.empty((), dtype=torch.int32, device=x.device).random_(0, 2**31)
    M = 4096
    N = 4096
    K = 256
    BLOCK_SIZE_M: tl.constexpr = 32
    BLOCK_SIZE_N: tl.constexpr = 64
    BLOCK_SIZE_K: tl.constexpr = 128
    GROUP_SIZE_M: tl.constexpr = 8
    if normalized_shape > K:
        raise ValueError("normalized_shape must be less than or equal to 2048")
    with torch.cuda.device(x.device.index):
        out = _bmm_rmsnorm_gelu_dropout_a(
            x,
            y,
            weight=None,
            ln_weight=torch.ones(normalized_shape, dtype=x.dtype, device=x.device),
            ln_bias=torch.zeros(normalized_shape, dtype=x.dtype, device=x.device),
            dropout_p=dropout_p,
            seed=seed,
            eps=eps,
            training=training,
            approximate=approximate,
            num_warps=4,
            num_stages=2,
        )
    return out
