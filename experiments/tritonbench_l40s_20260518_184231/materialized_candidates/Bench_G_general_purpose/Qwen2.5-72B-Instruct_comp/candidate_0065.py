import triton
import triton.language as tl

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, initial_state_ptr, 
    B, H, T, K, V, BK, BV, beta, scale, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = pid // (T * B)
    batch_id = (pid % (T * B)) // T
    seq_id = (pid % (T * B)) % T

    q_offset = (batch_id * H + head_id) * T * K + seq_id * K
    k_offset = (batch_id * H + head_id) * T * K
    v_offset = (batch_id * H + head_id) * T * V
    out_offset = (batch_id * H + head_id) * T * V + seq_id * V
    initial_state_offset = (batch_id * H + head_id) * V

    q = tl.load(q_ptr + q_offset, mask=seq_id < T, other=0.0)
    k = tl.load(k_ptr + k_offset, mask=seq_id < T, other=0.0)
    v = tl.load(v_ptr + v_offset, mask=seq_id < T, other=0.0)

    q = q * scale
    state = tl.load(initial_state_ptr + initial_state_offset, mask=seq_id == 0, other=0.0)

    for t in range(seq_id):
        k_t = tl.load(k_ptr + k_offset + t * K, mask=t < T, other=0.0)
        v_t = tl.load(v_ptr + v_offset + t * V, mask=t < T, other=0.0)
        state += q * k_t * beta + v_t

    tl.store(out_ptr + out_offset, state, mask=seq_id < T)

@triton.jit
def fused_recurrent_bwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, dout_ptr, initial_state_ptr, 
    dq_ptr, dk_ptr, dv_ptr, dinitial_state_ptr, 
    B, H, T, K, V, BK, BV, beta, scale, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = pid // (T * B)
    batch_id = (pid % (T * B)) // T
    seq_id = (pid % (T * B)) % T

    q_offset = (batch_id * H + head_id) * T * K + seq_id * K
    k_offset = (batch_id * H + head_id) * T * K
    v_offset = (batch_id * H + head_id) * T * V
    out_offset = (batch_id * H + head_id) * T * V + seq_id * V
    initial_state_offset = (batch_id * H + head_id) * V
    dout_offset = (batch_id * H + head_id) * T * V + seq_id * V

    q = tl.load(q_ptr + q_offset, mask=seq_id < T, other=0.0)
    k = tl.load(k_ptr + k_offset, mask=seq_id < T, other=0.0)
    v = tl.load(v_ptr + v_offset, mask=seq_id < T, other=0.0)
    dout = tl.load(dout_ptr + dout_offset, mask=seq_id < T, other=0.0)

    dq = tl.zeros((BLOCK_K,), dtype=tl.float32)
    dk = tl.zeros((BLOCK_K,), dtype=tl.float32)
    dv = tl.zeros((BLOCK_V,), dtype=tl.float32)
    dstate = tl.zeros((BLOCK_V,), dtype=tl.float32)

    for t in range(seq_id, -1, -1):
        k_t = tl.load(k_ptr + k_offset + t * K, mask=t < T, other=0.0)
        v_t = tl.load(v_ptr + v_offset + t * V, mask=t < T, other=0.0)
        dstate += dout * (q * k_t * beta + v_t)
        dq += dstate * k_t * beta
        dk += dstate * q * beta
        dv += dstate

    tl.store(dq_ptr + q_offset, dq, mask=seq_id < T)
    tl.store(dk_ptr + k_offset, dk, mask=seq_id < T)
    tl.store(dv_ptr + v_offset, dv, mask=seq_id < T)
    tl.store(dinitial_state_ptr + initial_state_offset, dstate, mask=seq_id == 0)

import torch
import triton
import triton.language as tl

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state, beta, scale, B, H, T, K, V, BK, BV, grid):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        initial_state = initial_state.contiguous()

        out = torch.empty_like(v)
        BLOCK_M, BLOCK_N, BLOCK_K = BK, BV, K

        fused_recurrent_fwd_kernel[
            grid
        ](q, k, v, out, initial_state, 
          B, H, T, K, V, BK, BV, beta, scale, 
          BLOCK_M, BLOCK_N, BLOCK_K)

        ctx.save_for_backward(q, k, v, out, initial_state)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BK, ctx.BV, ctx.beta, ctx.scale = B, H, T, K, V, BK, BV, beta, scale

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, initial_state = ctx.saved_tensors
        B, H, T, K, V, BK, BV, beta, scale = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BK, ctx.BV, ctx.beta, ctx.scale
        grid = ctx.grid

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dinitial_state = torch.zeros_like(initial_state)

        BLOCK_M, BLOCK_N, BLOCK_K = BK, BV, K

        fused_recurrent_bwd_kernel[
            grid
        ](q, k, v, out, dout, initial_state, 
          dq, dk, dv, dinitial_state, 
          B, H, T, K, V, BK, BV, beta, scale, 
          BLOCK_M, BLOCK_N, BLOCK_K)

        return dq, dk, dv, dinitial_state, None, None, None, None, None, None, None, None, None

def fused_recurrent_delta_rule(q, k, v, initial_state, beta=1.0, scale=1.0, B=None, H=None, T=None, K=None, V=None, BK=None, BV=None, grid=None):
    if B is None: B = q.shape[0]
    if H is None: H = q.shape[1]
    if T is None: T = q.shape[2]
    if K is None: K = q.shape[3]
    if V is None: V = v.shape[3]
    if BK is None: BK = 32
    if BV is None: BV = 32
    if grid is None: grid = (B * H * T,)

    return FusedRecurrentFunction.apply(q, k, v, initial_state, beta, scale, B, H, T, K, V, BK, BV, grid)

import torch

# Example tensors
B, H, T, K, V = 2, 4, 8, 16, 32
q = torch.randn(B, H, T, K, device='cuda')
k = torch.randn(B, H, T, K, device='cuda')
v = torch.randn(B, H, T, V, device='cuda')
initial_state = torch.randn(B, H, V, device='cuda')

# Forward pass
out = fused_recurrent_delta_rule(q, k, v, initial_state)

# Backward pass
out.sum().backward()
