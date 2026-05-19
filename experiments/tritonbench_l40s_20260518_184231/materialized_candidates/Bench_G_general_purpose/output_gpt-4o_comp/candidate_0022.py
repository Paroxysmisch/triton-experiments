import torch
import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(Q, K, V, Out, sm_scale, L, m, q_stride, k_stride, v_stride, o_stride, BLOCK: tl.constexpr):
    # Get the program id
    pid = tl.program_id(0)
    
    # Calculate the starting indices for Q, K, V
    q_offset = pid * BLOCK
    k_offset = 0
    v_offset = 0

    # Create pointers for the current block of Q, K, V
    q_tile_ptr = tl.make_block_ptr(Q, q_offset, q_stride, (BLOCK, BLOCK_DMODEL))
    k_tile_ptr = tl.make_block_ptr(K, k_offset, k_stride, (BLOCK_DMODEL, BLOCK_N))
    v_tile_ptr = tl.make_block_ptr(V, v_offset, v_stride, (BLOCK_N, BLOCK_DMODEL))

    # Load Q and K, compute QK^T
    q_tile = tl.load(q_tile_ptr)
    k_tile = tl.load(k_tile_ptr)
    qk = tl.dot(q_tile, k_tile)

    # Scale and apply softmax
    qk = qk * sm_scale
    qk = qk - tl.max(qk, axis=1, keepdim=True)
    qk_exp = tl.exp(qk)
    qk_softmax = qk_exp / tl.sum(qk_exp, axis=1, keepdim=True)

    # Store the normalization constant L and max m
    tl.store(L + pid, tl.sum(qk_exp, axis=1))
    tl.store(m + pid, tl.max(qk, axis=1))

    # Compute the output
    v_tile = tl.load(v_tile_ptr)
    out_tile = tl.dot(qk_softmax, v_tile)

    # Store the result
    out_tile_ptr = tl.make_block_ptr(Out, q_offset, o_stride, (BLOCK, BLOCK_DMODEL))
    tl.store(out_tile_ptr, out_tile)

@triton.jit
def _bwd_preprocess(DO, L, delta, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    do_offset = pid * BLOCK
    do_tile_ptr = tl.make_block_ptr(DO, do_offset, (BLOCK, BLOCK_DMODEL))

    # Load DO
    do_tile = tl.load(do_tile_ptr)

    # Scale by L and compute delta
    l_tile = tl.load(L + pid)
    delta_tile = do_tile / l_tile

    # Store delta
    delta_tile_ptr = tl.make_block_ptr(delta, do_offset, (BLOCK, BLOCK_DMODEL))
    tl.store(delta_tile_ptr, delta_tile)

@triton.jit
def _bwd_kernel(Q, K, V, DO, DQ, DK, DV, sm_scale, L, m, BLOCK: tl.constexpr):
    pid = tl.program_id(0)

    # Calculate offsets
    q_offset = pid * BLOCK
    k_offset = 0
    v_offset = 0

    # Load Q, K, V, DO
    q_tile_ptr = tl.make_block_ptr(Q, q_offset, (BLOCK, BLOCK_DMODEL))
    k_tile_ptr = tl.make_block_ptr(K, k_offset, (BLOCK_DMODEL, BLOCK_N))
    v_tile_ptr = tl.make_block_ptr(V, v_offset, (BLOCK_N, BLOCK_DMODEL))
    do_tile_ptr = tl.make_block_ptr(DO, q_offset, (BLOCK, BLOCK_DMODEL))

    q_tile = tl.load(q_tile_ptr)
    k_tile = tl.load(k_tile_ptr)
    v_tile = tl.load(v_tile_ptr)
    do_tile = tl.load(do_tile_ptr)

    # Recompute QK^T, apply softmax
    qk = tl.dot(q_tile, k_tile)
    qk = qk * sm_scale
    qk = qk - tl.max(qk, axis=1, keepdim=True)
    qk_exp = tl.exp(qk)
    qk_softmax = qk_exp / tl.sum(qk_exp, axis=1, keepdim=True)

    # Compute gradients
    dq_tile = tl.dot(do_tile, tl.transpose(v_tile))
    dv_tile = tl.dot(tl.transpose(qk_softmax), do_tile)
    dk_tile = tl.dot(tl.transpose(q_tile), do_tile)

    # Store gradients
    dq_tile_ptr = tl.make_block_ptr(DQ, q_offset, (BLOCK, BLOCK_DMODEL))
    dk_tile_ptr = tl.make_block_ptr(DK, k_offset, (BLOCK_DMODEL, BLOCK_N))
    dv_tile_ptr = tl.make_block_ptr(DV, v_offset, (BLOCK_N, BLOCK_DMODEL))

    tl.store(dq_tile_ptr, dq_tile)
    tl.store(dk_tile_ptr, dk_tile)
    tl.store(dv_tile_ptr, dv_tile)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale):
        BLOCK = 128

        # Allocate output tensors
        Out = torch.empty_like(Q)
        L = torch.empty(Q.shape[0] // BLOCK, dtype=Q.dtype, device=Q.device)
        m = torch.empty(Q.shape[0] // BLOCK, dtype=Q.dtype, device=Q.device)

        # Launch forward kernel
        grid = (Q.shape[0] // BLOCK,)
        _fwd_kernel[grid](Q, K, V, Out, sm_scale, L, m, Q.stride(), K.stride(), V.stride(), Out.stride(), BLOCK=BLOCK)

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, L, m)
        ctx.sm_scale = sm_scale

        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, L, m = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        BLOCK = 128

        # Allocate gradient tensors
        DQ = torch.empty_like(Q)
        DK = torch.empty_like(K)
        DV = torch.empty_like(V)

        # Preprocess DO
        delta = torch.empty_like(dOut)
        _bwd_preprocess[(Q.shape[0] // BLOCK,)](dOut, L, delta, BLOCK=BLOCK)

        # Launch backward kernel
        _bwd_kernel[(Q.shape[0] // BLOCK,)](Q, K, V, delta, DQ, DK, DV, sm_scale, L, m, BLOCK=BLOCK)

        return DQ, DK, DV, None

# Example usage
# Q, K, V are input tensors
# sm_scale is the scaling factor for softmax
# out = _attention.apply(Q, K, V, sm_scale)
