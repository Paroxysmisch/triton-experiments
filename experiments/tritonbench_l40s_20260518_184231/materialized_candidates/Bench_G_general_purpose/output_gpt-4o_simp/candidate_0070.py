import torch
import triton
import triton.language as tl

# Triton Kernel for Forward Pass: Update Hidden States
@triton.jit
def chunk_retention_fwd_kernel_h(q_ptr, k_ptr, v_ptr, h_ptr, out_ptr, scale, stride_q, stride_k, stride_v, stride_h, stride_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Load blocks of q, k, v
    q = tl.load(q_ptr + pid * stride_q + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * stride_k + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * stride_v + tl.arange(0, BLOCK_SIZE))
    # Compute new hidden state
    h = tl.load(h_ptr + pid * stride_h + tl.arange(0, BLOCK_SIZE))
    new_h = h * scale + tl.dot(q, k) * v
    tl.store(h_ptr + pid * stride_h + tl.arange(0, BLOCK_SIZE), new_h)
    # Write output
    tl.store(out_ptr + pid * stride_out + tl.arange(0, BLOCK_SIZE), new_h)

# Triton Kernel for Forward Pass: Compute Output
@triton.jit
def chunk_retention_fwd_kernel_o(q_ptr, k_ptr, v_ptr, out_ptr, scale, stride_q, stride_k, stride_v, stride_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Load blocks of q, k, v
    q = tl.load(q_ptr + pid * stride_q + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * stride_k + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * stride_v + tl.arange(0, BLOCK_SIZE))
    # Compute output
    out = tl.dot(q, k) * v * scale
    tl.store(out_ptr + pid * stride_out + tl.arange(0, BLOCK_SIZE), out)

# Triton Kernel for Backward Pass: Gradients for Hidden State
@triton.jit
def chunk_retention_bwd_kernel_dh(dout_ptr, dh_ptr, q_ptr, k_ptr, v_ptr, scale, stride_dout, stride_dh, stride_q, stride_k, stride_v, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Load blocks
    dout = tl.load(dout_ptr + pid * stride_dout + tl.arange(0, BLOCK_SIZE))
    q = tl.load(q_ptr + pid * stride_q + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * stride_k + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * stride_v + tl.arange(0, BLOCK_SIZE))
    # Compute gradient for hidden state
    dh = dout * scale * tl.dot(q, k) * v
    tl.store(dh_ptr + pid * stride_dh + tl.arange(0, BLOCK_SIZE), dh)

# Triton Kernel for Backward Pass: Gradients for Input Tensors
@triton.jit
def chunk_retention_bwd_kernel_dqkv(dout_ptr, dq_ptr, dk_ptr, dv_ptr, q_ptr, k_ptr, v_ptr, scale, stride_dout, stride_dq, stride_dk, stride_dv, stride_q, stride_k, stride_v, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Load blocks
    dout = tl.load(dout_ptr + pid * stride_dout + tl.arange(0, BLOCK_SIZE))
    q = tl.load(q_ptr + pid * stride_q + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * stride_k + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * stride_v + tl.arange(0, BLOCK_SIZE))
    # Compute gradients
    dq = dout * scale * tl.dot(k, v)
    dk = dout * scale * tl.dot(q, v)
    dv = dout * scale * tl.dot(q, k)
    # Store gradients
    tl.store(dq_ptr + pid * stride_dq + tl.arange(0, BLOCK_SIZE), dq)
    tl.store(dk_ptr + pid * stride_dk + tl.arange(0, BLOCK_SIZE), dk)
    tl.store(dv_ptr + pid * stride_dv + tl.arange(0, BLOCK_SIZE), dv)

# Wrapper Functions
def chunk_fwd_h_fn(q, k, v, h, scale, BLOCK_SIZE=128):
    grid = (q.size(0) // BLOCK_SIZE,)
    out = torch.empty_like(h)
    chunk_retention_fwd_kernel_h[grid](q, k, v, h, out, scale, q.stride(0), k.stride(0), v.stride(0), h.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out

def chunk_fwd_o_fn(q, k, v, scale, BLOCK_SIZE=128):
    grid = (q.size(0) // BLOCK_SIZE,)
    out = torch.empty_like(q)
    chunk_retention_fwd_kernel_o[grid](q, k, v, out, scale, q.stride(0), k.stride(0), v.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out

def chunk_bwd_dh_fn(dout, q, k, v, scale, BLOCK_SIZE=128):
    grid = (dout.size(0) // BLOCK_SIZE,)
    dh = torch.empty_like(dout)
    chunk_retention_bwd_kernel_dh[grid](dout, dh, q, k, v, scale, dout.stride(0), dh.stride(0), q.stride(0), k.stride(0), v.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return dh

def chunk_bwd_dqkv_fn(dout, q, k, v, scale, BLOCK_SIZE=128):
    grid = (dout.size(0) // BLOCK_SIZE,)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    chunk_retention_bwd_kernel_dqkv[grid](dout, dq, dk, dv, q, k, v, scale, dout.stride(0), dq.stride(0), dk.stride(0), dv.stride(0), q.stride(0), k.stride(0), v.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return dq, dk, dv

# PyTorch Custom Autograd Function
class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_h=None, scale=1.0):
        if initial_h is None:
            initial_h = torch.zeros_like(q)
        ctx.save_for_backward(q, k, v, initial_h)
        ctx.scale = scale
        h = chunk_fwd_h_fn(q, k, v, initial_h, scale)
        out = chunk_fwd_o_fn(q, k, v, scale)
        return out, h

    @staticmethod
    def backward(ctx, dout, dh):
        q, k, v, initial_h = ctx.saved_tensors
        scale = ctx.scale
        dq, dk, dv = chunk_bwd_dqkv_fn(dout, q, k, v, scale)
        dh = chunk_bwd_dh_fn(dout, q, k, v, scale)
        return dq, dk, dv, dh, None

# Main Entry Point
def chunk_retention(q, k, v, initial_h=None, scale=1.0):
    return ChunkRetentionFunction.apply(q, k, v, initial_h, scale)
