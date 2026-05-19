import torch
import triton
import triton.language as tl


@triton.heuristics({
    "IS_FP16": lambda **meta: meta["x_dtype"] in [torch.float16]
})
@triton.jit
def _softmax(
    output_ptr,      # *f32 / *f16
    input_ptr,       # *f32 / *f16
    mask_ptr,        # *f8 (optional, could store byte mask)
    B, M, N,         # dimensions
    stride_inB,      # stride for batch dimension of input
    stride_inM,      # stride for M dimension of input
    stride_inN,      # stride for N dimension of input
    stride_outB,     # stride for batch dimension of output
    stride_outM,     # stride for M dimension of output
    stride_outN,     # stride for N dimension of output
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr
):
    # program_id_x will iterate over B * M
    pid = tl.program_id(0)
    b_idx = pid // M
    m_idx = pid % M

    # row pointers
    row_in_ptr = input_ptr + b_idx * stride_inB + m_idx * stride_inM
    row_out_ptr = output_ptr + b_idx * stride_outB + m_idx * stride_outM
    if mask_ptr != 0:
        mask_row_ptr = mask_ptr + b_idx * M * N + m_idx * N

    # Compute indexes
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # Load
    row_val = tl.load(row_in_ptr + idx * stride_inN, mask=mask, other=-float('inf'))

    # Apply causal mask if needed
    if CAUSAL:
        # For demonstration, assume a trivial causal dimension based on m_idx
        # Typically you have a separate dimension for sequence length
        # which is used for the mask
        causal_mask = idx > m_idx
        row_val = tl.where(causal_mask, float('-inf'), row_val)

    # Apply external mask if needed
    if MASK_TYPE == 1 and mask_ptr != 0:
        # mask_ptr is expected to be 0 or 1, or any float for some custom masking
        row_mask_val = tl.load(mask_row_ptr + idx, mask=mask, other=1.0)
        row_val = tl.where(row_mask_val == 0.0, float('-inf'), row_val)

    # Sub-max
    local_max = tl.max(row_val, axis=0)
    row_val = row_val - local_max

    # Exponential
    numerator = tl.exp(row_val)
    denominator = tl.sum(numerator, axis=0)
    softmax_val = numerator / denominator

    if LOG:
        # log-softmax
        softmax_val = row_val - tl.log(denominator)

    # Store
    tl.store(row_out_ptr + idx * stride_outN, softmax_val, mask=mask)


def softmax(x, mask=None, log=False, causal=False, mask_type=0):
    assert x.ndim == 3, "Input tensor must be 3D"
    B, M, N = x.shape
    x_dtype = x.dtype

    # Convert mask to a pointer if it's provided
    mask_ptr = 0
    if mask is not None:
        assert mask.shape == (B, M, N), "Mask shape must match input shape"
        mask = mask.to(x.device)
        mask_ptr = mask.data_ptr()

    # Allocate output
    out = torch.empty_like(x)

    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (B * M,)
    _softmax[grid](
        out, x, mask_ptr,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        LOG=log,
        CAUSAL=causal,
        MASK_TYPE=mask_type,
        BLOCK_SIZE=BLOCK_SIZE,
        x_dtype=x_dtype
    )
    return out


@triton.heuristics({
    "IS_FP16": lambda **meta: meta["grad_dtype"] in [torch.float16]
})
@triton.jit
def _softmax_backward(
    grad_in_ptr,       # *f32 / *f16
    grad_out_ptr,      # *f32 / *f16
    output_ptr,        # *f32 / *f16 (softmax output)
    mask_ptr,          # *f8
    B, M, N,
    stride_gradB,
    stride_gradM,
    stride_gradN,
    stride_outB,
    stride_outM,
    stride_outN,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = pid // M
    m_idx = pid % M

    grad_row_in_ptr = grad_in_ptr + b_idx * stride_gradB + m_idx * stride_gradM
    grad_row_out_ptr = grad_out_ptr + b_idx * stride_gradB + m_idx * stride_gradM
    out_row_ptr = output_ptr + b_idx * stride_outB + m_idx * stride_outM
    if mask_ptr != 0:
        mask_row_ptr = mask_ptr + b_idx * M * N + m_idx * N

    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    grad_val = tl.load(grad_row_in_ptr + idx * stride_gradN, mask=mask, other=0.0)
    out_val = tl.load(out_row_ptr + idx * stride_outN, mask=mask, other=0.0)

    if CAUSAL:
        causal_mask = idx > m_idx
        grad_val = tl.where(causal_mask, 0.0, grad_val)

    if MASK_TYPE == 1 and mask_ptr != 0:
        row_mask_val = tl.load(mask_row_ptr + idx, mask=mask, other=1.0)
        grad_val = tl.where(row_mask_val == 0.0, 0.0, grad_val)

    if not LOG:
        # Normal softmax backward
        dot = tl.sum(grad_val * out_val, axis=0)
        grad_val = out_val * (grad_val - dot)
    else:
        # Log-softmax backward
        sum_grad = tl.sum(grad_val, axis=0)
        grad_val = grad_val - tl.exp(-out_val) * sum_grad

    tl.store(grad_row_out_ptr + idx * stride_gradN, grad_val, mask=mask)


def softmax_backward(grad, out, mask=None, log=False, causal=False, mask_type=0):
    assert grad.ndim == 3 and out.ndim == 3, "Gradient and output must be 3D"
    B, M, N = grad.shape
    grad_dtype = grad.dtype

    mask_ptr = 0
    if mask is not None:
        assert mask.shape == (B, M, N), "Mask shape must match input shape"
        mask = mask.to(grad.device)
        mask_ptr = mask.data_ptr()

    grad_out = torch.empty_like(grad)

    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (B * M,)
    _softmax_backward[grid](
        grad, grad_out, out, mask_ptr,
        B, M, N,
        grad.stride(0), grad.stride(1), grad.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        LOG=log,
        CAUSAL=causal,
        MASK_TYPE=mask_type,
        BLOCK_SIZE=BLOCK_SIZE,
        grad_dtype=grad_dtype
    )
    return grad_out
