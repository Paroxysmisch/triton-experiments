import torch
import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q_ptr,
    K_ptr,
    M_ptr,
    Out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
):

    row_block_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    # Compute the start indices of the current block in the grid.
    row_start = row_block_idx * BLOCK_M
    col_start = col_block_idx * BLOCK_N

    # Load the blocks of Q and K to registers.
    Q_block = tl.load(Q_ptr + row_start + col_start * Q_ptr.strides[0])
    K_block = tl.load(K_ptr + row_start + col_start * K_ptr.strides[0])

    # Compute Q @ K.
    qk = tl.dot(Q_block, K_block)

    # Scale the result by sm_scale.
    qk_scaled = qk * SM_SCALE

    # Load the mask block to a register.
    M_block = tl.load(M_ptr + row_start + col_start * M_ptr.strides[0])

    # Compute the attention scores by masking out invalid positions in the mask block.
    o = tl.where(M_block, qk_scaled, -1e10)

    # Store the computed scores back to DRAM.
    tl.store(Out_ptr + row_start + col_start * Out_ptr.strides[0], o)


def get_score(Q, K, M, SM_SCALE, BLOCK_M=32, BLOCK_N=32):
    device = Q.device

    # Allocate output tensor.
    Out = torch.empty_like(M)

    # Calculate grid size based on Q and K.
    grid = lambda META: (
        triton.cdiv(Q.shape[0], META["BLOCK_M"]),
        triton.cdiv(K.shape[1], META["BLOCK_N"]),
    )

    # Run the kernel.
    try:
        _score_kernel[grid](Q, K, M, Out, SM_SCALE=SM_SCALE)
    except triton.TritonException_ResourceConstraint as e:
        # If we encounter a resource constraint error, reduce BLOCK_M and BLOCK_N by half
        # and retry the execution.
        BLOCK_M //= 2
        BLOCK_N //= 2
        return get_score(Q, K, M, SM_SCALE, BLOCK_M, BLOCK_N)

    return Out
