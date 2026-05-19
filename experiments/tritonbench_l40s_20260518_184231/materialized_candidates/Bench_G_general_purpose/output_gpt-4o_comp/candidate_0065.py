import triton
import triton.language as tl

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, initial_state_ptr, beta, scale, 
    B, H, T, K, V, 
    BK, BV, 
    **meta
):
    # Compute the grid dimensions
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Compute the starting index for this block
    q_offset = batch_idx * H * T * K + head_idx * T * K + seq_idx * K
    k_offset = batch_idx * H * T * K + head_idx * T * K + seq_idx * K
    v_offset = batch_idx * H * T * V + head_idx * T * V + seq_idx * V
    out_offset = batch_idx * H * T * V + head_idx * T * V + seq_idx * V

    # Load data from global memory
    q = tl.load(q_ptr + q_offset, mask=seq_idx < T)
    k = tl.load(k_ptr + k_offset, mask=seq_idx < T)
    v = tl.load(v_ptr + v_offset, mask=seq_idx < T)

    # Initialize hidden state
    if initial_state_ptr is not None:
        state = tl.load(initial_state_ptr + out_offset, mask=seq_idx < T)
    else:
        state = tl.zeros([BK], dtype=tl.float32)

    # Perform element-wise multiplication and accumulation
    for i in range(BK):
        weighted_q = q[i] * beta
        state = state * scale + weighted_q * k[i]
        state = state + v[i]

    # Store the output
    tl.store(out_ptr + out_offset, state, mask=seq_idx < T)

#### Backward Kernel


import torch

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, initial_state=None, scale=1.0):
        B, H, T, K = q.shape
        _, _, _, V = v.shape
        BK, BV = 128, 128  # Example block sizes
        grid = (B, H, T)

        # Allocate output
        out = torch.empty_like(v)

        # Launch forward kernel
        fused_recurrent_fwd_kernel[grid](
            q, k, v, out, initial_state, beta, scale,
            B, H, T, K, V, BK, BV
        )

        # Save context for backward
        ctx.save_for_backward(q, k, v, out)
        ctx.beta = beta
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out = ctx.saved_tensors
        B, H, T, K = q.shape
        _, _, _, V = v.shape
        BK, BV = 128, 128  # Example block sizes
        grid = (B, H, T)

        # Allocate gradients
        q_grad = torch.empty_like(q)
        k_grad = torch.empty_like(k)
        v_grad = torch.empty_like(v)

        # Launch backward kernel
        fused_recurrent_bwd_kernel[grid](
            q, k, v, grad_output, q_grad, k_grad, v_grad,
            B, H, T, K, V, BK, BV
        )

        return q_grad, k_grad, v_grad, None, None, None

def fused_recurrent_delta_rule(q, k, v, beta, initial_state=None, scale=1.0):
    return FusedRecurrentFunction.apply(q, k, v, beta, initial_state, scale)
