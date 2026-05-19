import torch
import triton
import triton.language as tl

@triton.jit
def _elementwise_mul_add_kernel(
    x1_ptr, x2_ptr, other_ptr, out_ptr,
    N, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    o = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    # Z = X1 * X2
    z = x1 * x2
    # S = Z + other
    s = z + o

    tl.store(out_ptr + offsets, s, mask=mask)

@triton.jit
def _log_softmax_lastdim_kernel(
    data_ptr, out_ptr,
    rows, cols,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    row_id = tl.program_id(0)
    # Each program handles one row
    if row_id >= rows:
        return

    row_start = row_id * cols
    # Load row into SRAM
    offsets = tl.arange(0, BLOCK_N)
    ptrs = data_ptr + (row_start + offsets)
    mask = offsets < cols
    row_val = tl.load(ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row_val, axis=0)
    row_val = row_val - row_max
    exp_val = tl.exp(row_val)
    denom = tl.sum(exp_val, axis=0)
    # log_softmax
    out_val = row_val - tl.log(denom)
    # Store
    tl.store(out_ptr + row_start + offsets, out_val, mask=mask)

@triton.jit
def _dropout_kernel(
    x_ptr, out_ptr,
    rand_ptr,
    N,
    p, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    r = tl.load(rand_ptr + offsets, mask=mask, other=1.0)

    keep_mask = r > p
    # Zero out elements based on dropout mask
    dropped = tl.where(keep_mask, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, dropped, mask=mask)

@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,  # A shape = [B, M, K], B shape = [B, K, N], C shape = [B, M, N]
    stride_aB, stride_aM, stride_aK,
    stride_bB, stride_bK, stride_bN,
    stride_cB, stride_cM, stride_cN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Batch id:
    bid = tl.program_id(0)
    # Coordinates for block of C
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    pm = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    pn = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointer arithmetic for each batch
    a_off = bid * stride_aB
    b_off = bid * stride_bB
    c_off = bid * stride_cB

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_block in range(0, K, BLOCK_K):
        pk = k_block + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + a_off + (pm[:, None] * stride_aM + pk[None, :] * stride_aK)
        b_ptrs = b_ptr + b_off + (pk[:, None] * stride_bK + pn[None, :] * stride_bN)

        a_vals = tl.load(a_ptrs, mask=(pm[:, None] < M) & (pk[None, :] < K), other=0.0)
        b_vals = tl.load(b_ptrs, mask=(pk[:, None] < K) & (pn[None, :] < N), other=0.0)
        acc += tl.dot(a_vals, b_vals)

    c_ptrs = c_ptr + c_off + (pm[:, None] * stride_cM + pn[None, :] * stride_cN)
    tl.store(c_ptrs, acc, mask=(pm[:, None] < M) & (pn[None, :] < N))

def fused_mul_add_logsoftmax_dropout_bmm(
    input1, input2, other, mat2,
    p=0.5, training=True, inplace=False, dim=-1, *, out=None
):
    """
    fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2,
                                         p=0.5, training=True, inplace=False, dim=-1, *, out=None) -> Tensor

    Computes:
        Z = input1 * input2
        S = Z + other
        L = log_softmax(S, dim)
        D = Dropout(L, p)         [only applied if training=True]
        Y = bmm(D, mat2)
    """
    # Broadcast shapes for (input1 * input2) + other
    x1, x2 = torch.broadcast_tensors(input1, input2)
    x1x2_shape = x1.shape
    # Also broadcast 'other' to same shape
    if other.shape != x1x2_shape:
        other_b = other.expand(x1x2_shape)
    else:
        other_b = other

    #
