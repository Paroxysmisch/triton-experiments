import torch
import triton
import triton.language as tl

# Define Triton kernel for forward pass
@triton.jit
def parallel_rebased_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, z_ptr, scores_ptr,
    q_stride, k_stride, v_stride, o_stride, z_stride, scores_stride,
    n, bsz, h, seq_len, d_model, BTL, BTS, BK, BV, use_scale, use_normalize
):
    pid = tl.program_id(axis=0)
    block_idx = pid // (BTL * BTS * BK * BV)
    block_size = BTL * BTS * BK * BV
    q_offset = block_idx * n * seq_len * d_model
    k_offset = block_idx * n * seq_len * d_model
    v_offset = block_idx * n * seq_len * d_model
    o_offset = block_idx * n * seq_len * d_model
    z_offset = block_idx * n * seq_len * d_model
    scores_offset = block_idx * n * seq_len * seq_len

    q_block = tl.load(q_ptr + q_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    k_block = tl.load(k_ptr + k_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    v_block = tl.load(v_ptr + v_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)

    scores = tl.dot(q_block, k_block, trans_b=True)
    if use_scale:
        scores /= tl.sqrt(tl.constant(d_model, dtype=tl.float32))

    tl.store(scores_ptr + scores_offset, scores, mask=tl.arange(0, seq_len) < seq_len)

    if use_normalize:
        z_block = tl.sum(scores, axis=1, keepdim=True)
        z_block = 1.0 / tl.sqrt(z_block)
        tl.store(z_ptr + z_offset, z_block, mask=tl.arange(0, seq_len) < seq_len)

    o_block = tl.dot(scores, v_block)
    tl.store(o_ptr + o_offset, o_block, mask=tl.arange(0, seq_len) < seq_len)

# Define Triton kernel for backward pass
@triton.jit
def parallel_rebased_bwd_kernel(
    do_ptr, dz_ptr, q_ptr, k_ptr, v_ptr, scores_ptr, o_ptr,
    do_stride, dz_stride, q_stride, k_stride, v_stride, o_stride,
    n, bsz, h, seq_len, d_model, BTL, BTS, BK, BV, use_scale, use_normalize
):
    pid = tl.program_id(axis=0)
    block_idx = pid // (BTL * BTS * BK * BV)
    block_size = BTL * BTS * BK * BV
    do_offset = block_idx * n * seq_len * d_model
    dz_offset = block_idx * n * seq_len * d_model
    q_offset = block_idx * n * seq_len * d_model
    k_offset = block_idx * n * seq_len * d_model
    v_offset = block_idx * n * seq_len * d_model
    o_offset = block_idx * n * seq_len * d_model
    scores_offset = block_idx * n * seq_len * seq_len

    do_block = tl.load(do_ptr + do_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    dz_block = tl.load(dz_ptr + dz_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    q_block = tl.load(q_ptr + q_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    k_block = tl.load(k_ptr + k_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    v_block = tl.load(v_ptr + v_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    scores_block = tl.load(scores_ptr + scores_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)
    o_block = tl.load(o_ptr + o_offset, mask=tl.arange(0, seq_len) < seq_len, other=0.0)

    # Compute gradients for queries (dq)
    dq_block = tl.dot(scores_block, do_block, trans_a=True)
    tl.store(q_ptr + q_offset, dq_block, mask=tl.arange(0, seq_len) < seq_len)

    # Compute gradients for key-value pairs (dk, dv)
    dk_block = tl.dot(do_block, v_block, trans_b=True)
    dv_block = tl.dot(scores_block, do_block)
    tl.store(k_ptr + k_offset, dk_block, mask=tl.arange(0, seq_len) < seq_len)
    tl.store(v_ptr + v_offset, dv_block, mask=tl.arange(0, seq_len) < seq_len)

# Define the custom operation class
class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_scale=False, use_normalize=False):
        n, bsz, h, seq_len, d_model = q.shape
        assert d_model <= 128, "Feature dimension must be less than or equal to 128"
        o = torch.empty_like(q)
        z = torch.empty_like(q) if use_normalize else None
        scores = torch.empty_like(q)
        grid = (triton.cdiv(n * seq_len, BTL) * triton.cdiv(seq_len, BTS) * triton.cdiv(seq_len, BK) * triton.cdiv(d_model, BV),)
        parallel_rebased_fwd_kernel[grid](q, k, v, o, z, scores, q.stride(0), k.stride(0), v.stride(0), o.stride(0), z.stride(0), scores.stride(0), n, bsz, h, seq_len, d_model, BTL, BTS, BK, BV, use_scale, use_normalize)
        ctx.save_for_backward(q, k, v, o, z, scores)
        return o, z if use_normalize else o

    @staticmethod
    def backward(ctx, do, dz):
        q, k, v, o, z, scores = ctx.saved_tensors
        n, bsz, h, seq_len, d_model = q.shape
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dz_block = dz if dz is not None else torch.zeros_like(z)
        grid = (triton.cdiv(n * seq_len, BTL) * triton.cdiv(seq_len, BTS) * triton.cdiv(seq_len, BK) * triton.cdiv(d_model, BV),)
        parallel_rebased_bwd_kernel[grid](do, dz_block, q, k, v, scores, o, do.stride(0), dz_block.stride(0), q.stride(0), k.stride(0), v.stride(0), o.stride(0), n, bsz, h, seq_len, d_model, BTL, BTS, BK, BV, False, dz is not None)
        return dq, dk, dv, None, None

# User-facing API
def parallel_rebased(q, k, v, use_scale=False, use_normalize=False, return_both=False):
    o, z = ParallelBasedFunction.apply(q, k, v, use_scale, use_normalize)
    return (o, z) if return_both else o
