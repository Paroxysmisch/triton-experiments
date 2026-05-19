import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Optional helper: simple autotune configs for elementwise kernels
# ----------------------------------------------------------------------
def element_wise_kernel_configs():
    # You can adjust block sizes or add more configs if desired
    return [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
    ]

# ----------------------------------------------------------------------
# Dropout kernels (as in Document 1)
# ----------------------------------------------------------------------
@triton.jit
def apply_dropout(x, drop_p, seed, offset):
    r = tl.rand(seed, offset)
    # keep_prob = 1 - drop_p
    # if r < drop_p, set to 0, else scale by 1/(1-drop_p)
    return tl.where(r < drop_p, 0, x / (1 - drop_p))

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_forward_kernel(
    input_ptr, output_ptr, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out = apply_dropout(x, drop_p, seed, offsets)
    tl.store(output_ptr + offsets, out, mask=mask)

# ----------------------------------------------------------------------
# GELU kernel
# ----------------------------------------------------------------------
@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def gelu_forward_kernel(
    x_ptr, out_ptr, size,
    approximate,  # 0 for 'none', 1 for 'tanh'
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # approximate GELU ('tanh'): 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    # We'll branch on approximate:
    is_tanh_approx = approximate == 1
    # Using separate computations:
    x_sq = x * x
    x_cb = x_sq * x

    # 1 / sqrt(2)
    inv_sqrt2 = 0.70710678118
    # sqrt(2/pi)
    coef_tanh = 0.7978845608
    # 0.044715
    coef_cb    = 0.044715

    # exact formula
    gelu_exact = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    # tanh-based approximate
    tanh_approx_input = coef_tanh * (x + coef_cb * x_cb)
    gelu_tanh_approx = 0.5 * x * (1.0 + tl.tanh(tanh_approx_input))

    out = tl.where
