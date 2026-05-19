import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    stride_input1_b, stride_input1_n, stride_input1_m,
    stride_input2_b, stride_input2_m, stride_input2_p,
    stride_output_b, stride_output_n, stride_output_p,
    p, use_dropout: tl.constexpr,
    seed, offset,
    gelu_approximate: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    # Batch index
    b = tl.program_id(0)
    # Row indices for input1 and output
    n_idx = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Column indices for input2 and output
    p_idx = tl.program_id(2) * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)

    # Masks to handle boundary conditions
    n_mask = n_idx < N
    p_mask = p_idx < P

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)

    # Loop over M dimension to compute block matrix multiplication
    for m in range(0, M, BLOCK_SIZE_M):
        m_offs = m + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offs < M

        # Load input1 block [BLOCK_SIZE_N, BLOCK_SIZE_M]
        input1_ptrs = input1_ptr + b * stride_input1_b + n_idx[:, None] * stride_input1_n + m_offs[None, :] * stride_input1_m
        a = tl.load(input1_ptrs, mask=n_mask[:, None] & m_mask[None, :], other=0.0)
        a = a.to(tl.float32)

        # Load input2 block [BLOCK_SIZE_M, BLOCK_SIZE_P]
        input2_ptrs = input2_ptr + b * stride_input2_b + m_offs[:, None] * stride_input2_m + p_idx[None, :] * stride_input2_p
        b_vals = tl.load(input2_ptrs, mask=m_mask[:, None] & p_mask[None, :], other=0.0)
        b_vals = b_vals.to(tl.float32)

        # Compute matrix multiplication
        acc += tl.dot(a, b_vals)

    # Apply dropout if training and p > 0
    if use_dropout:
        # Calculate element IDs for RNG
        n_exp = tl.expand_dims(n_idx, 1)
        p_exp = tl.expand_dims(p_idx, 0)
        element_ids = b * N * P + n_exp * P + p_exp
        # Generate random numbers
        rand = tl.rand(seed, element_ids)
        # Create dropout mask and scale
        mask = rand > p
        scale = 1.0 / (1.0 - p)
        acc = acc * mask * scale

    # Apply GELU activation
    if gelu_approximate == 'tanh':
        # Approximate GELU with tanh
        gelu = acc * 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.math.pi) * (acc + 0.044715 * acc * acc * acc)))
    else:
        # Exact GELU using erf
        gelu = 0.5 * acc * (1.0 + tl.erf(acc / tl.sqrt(2.0)))

    # Convert back to original data type
    gelu = gelu.to(input1_ptr.dtype.element_ty)

    # Write output
    output_ptrs = output_ptr + b * stride_output_b + n_idx[:, None] * stride_output_n + p_idx[None, :] * stride_output_p
    tl.store(output_ptrs, gelu, mask=n_mask[:, None] & p_mask[None, :])

def fused_bmm_dropout_gelu(
    input1: torch.Tensor,
    input2: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    approximate: str = 'none',
    *,
    out: torch.Tensor = None
) -> torch.Tensor:
    # Check input shapes
    assert input1.dim() == 3 and input2.dim() == 3, "Inputs must be 3D tensors"
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, f"Batch size and M dimension must match. Got input1: {input1.shape}, input2: {input2.shape}"

    # Compute output shape
    output_shape = (B, N, P)
    
    # Handle inplace and out tensor
    if out is not None:
        assert out.shape == output_shape, f"out tensor has wrong shape: expected {output_shape}, got {out.shape}"
        if not inplace:
            assert not torch.is_grad_enabled() or (out.data_ptr() != input1.data_ptr() and out.data_ptr() != input2.data_ptr()), \
                "out tensor must not alias inputs when inplace=False"
    else:
        if inplace:
            raise ValueError("inplace=True requires an out tensor to be provided")
        out = torch.empty(output_shape, device=input1.device, dtype=input1.dtype)
    
    # Determine if dropout is applied
    use_dropout = training and p > 0.0

    # Seed and offset for RNG (simplified, should use PyTorch's RNG state in practice)
    seed = 0
    offset = 0
    if use_dropout:
        # Generate random seed and offset (placeholder)
        max_seed = 2**32 - 1
        seed = torch.randint(0, max_seed, (1,), device=input1.device).item()
        offset = 0  # Offset handling is simplified; actual implementation needs proper counter management

    # GELU approximation
    gelu_approximate_tl = 'tanh' if approximate == 'tanh' else 'none'

    # Tuning parameters for kernel
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_P = 16

    # Grid configuration
    grid = (
        B,
        triton.cdiv(N, BLOCK_SIZE_N),
        triton.cdiv(P, BLOCK_SIZE_P),
    )

    # Launch kernel
    fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        p, use_dropout,
        seed, offset,
        gelu_approximate_tl,
        BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_P,
    )

    return out
