import triton
import triton.language as tl
import torch

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    A_ptr,            # ptr to A
    V_ptr,            # ptr to vec
    OTHER_ptr,        # ptr to other
    O_ptr,            # ptr to output
    N,                # number of rows in A
    M,                # number of cols in A
    strideAN,         # stride for rows of A
    strideAM,         # stride for cols of A
    strideO,          # stride for out
    strideOTHER,      # stride for other (if tensor)
    alpha,            # scalar alpha
    other_numel,      # numel for 'other'
    BLOCK_SIZE_M: tl.constexpr
):
    row_id = tl.program_id(0)
    # Each program handles one row of A if row_id < N
    if row_id >= N:
        return

    # Pointer offsets for this row
    A_row_ptr = A_ptr + row_id * strideAN
    O_row_ptr = O_ptr + row_id * strideO

    # We'll compute dot(A[row_id, :], V[:]) in chunks
    # Accumulate partial sums in fp32 for numerical stability
    acc = tl.zeros([], dtype=tl.float32)
    # Process columns in steps of BLOCK_SIZE_M
    for col_offset in range(0, M, BLOCK_SIZE_M):
        cols = tl.arange(0, BLOCK_SIZE_M)
        cols_mask = cols + col_offset < M

        # Load A row chunk
        a_data = tl.load(
            A_row_ptr + (cols + col_offset) * strideAM,
            mask=cols_mask,
            other=0.0
        )
        # Load corresponding chunk from vec
        v_data = tl.load(
            V_ptr + (cols + col_offset),
            mask=cols_mask,
            other=0.0
        )

        # Accumulate dot product
        acc += tl.sum(a_data.to(tl.float32) * v_data.to(tl.float32))

    # Apply sigmoid
    # z = dot product, s = 1 / (1 + exp(-z))
    z_fp32 = acc
    s_fp32 = 1.0 / (1.0 + tl.exp(-z_fp32))
    s = s_fp32.to(tl.float32)

    # Broadcast 'other' if needed
    # Case 1: other_numel == 1 -> single scalar
    # Case 2: other_numel == N -> per row
    # all others are not valid for this fused kernel
    if other_numel == 1:
        b_val = tl.load(OTHER_ptr)
    else:
        # other is assumed broadcastable with dimension N
        b_val = tl.load(OTHER_ptr + row_id * strideOTHER)
    # Subtract alpha * other
    out_val = s - alpha * b_val

    # Store result
    tl.store(O_row_ptr, out_val)


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    """
    fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None) -> Tensor
    input : Tensor of shape (n, m)
    vec   : Tensor of shape (m)
    other : Tensor or scalar (broadcastable to shape (n,) or a single scalar)
    alpha : scalar, default=1
    out   : optional output tensor
    """
    # Shape checks
    if input.dim() != 2:
        raise ValueError("input must be 2D with shape (n, m).")
    if vec.dim() != 1:
        raise ValueError("vec must be 1D with shape (m,).")
    n, m = input.shape
    if vec.shape[0] != m:
        raise ValueError("vec's length must match input.shape[1].")
    
    # Convert other to tensor if scalar
    if isinstance(other, (int, float)):
        other_t = torch.tensor([other], dtype=input.dtype, device=input.device)
    else:
        other_t = other.to(input.device, input.dtype)
        # Basic broadcast check: either shape==[n], shape==[1], or can be size 1
        if other_t.numel() not in (1, n):
            raise ValueError("other must be broadcastable to shape (n,) or a single scalar.")

    # Prepare output
    if out is None:
        out = torch.empty((n,), dtype=input.dtype, device=input.device)

    # Ensure contiguous
    A_contig = input.contiguous()
    V_contig = vec.contiguous()
    OTHER_contig = other_t.contiguous()
    OUT_contig = out.contiguous()

    # Strides
    # A shape (n, m): strideAN = m, strideAM = 1 if contiguous in row-major
    strideAN = A_contig.stride(0)
    strideAM = A_contig.stride(1)
    strideO = OUT_contig.stride(0)
    strideOTHER = 0
    if OTHER_contig.numel() == n:
        strideOTHER = OTHER_contig.stride(0)

    # Kernel launch
    grid = (n,)
    block_size_m = 128  # chunk over columns in blocks of 128

    # Launch kernel
    _fused_mv_sigmoid_sub_kernel[grid](
        A_contig, 
        V_contig,
        OTHER_contig,
        OUT_contig,
        n, 
        m,
        strideAN, 
        strideAM, 
        strideO,
        strideOTHER,
        alpha, 
        OTHER_contig.numel(),
        BLOCK_SIZE_M=block_size_m
    )

    # Return out if not in-place
    if out is not OUT_contig:
        out.copy_(OUT_contig)
    return out
