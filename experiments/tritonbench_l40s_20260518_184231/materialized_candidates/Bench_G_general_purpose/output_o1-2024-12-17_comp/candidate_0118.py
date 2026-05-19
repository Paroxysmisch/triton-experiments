import torch
import triton
import triton.language as tl


@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    Q_PTR, K_PTR, V_PTR, W_PTR, U_PTR,
    OUT_PTR, FINAL_H_PTR,
    T, D,
    USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE,
    SCALE,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load initial hidden state if needed
    hidden_state = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if USE_INITIAL_STATE != 0:
        hidden_state = tl.load(FINAL_H_PTR + offs, mask=mask, other=hidden_state)

    # Recurrent loop
    rng = range(T - 1, -1, -1) if REVERSE != 0 else range(T)
    for t in rng:
        q_off = t * D + offs
        k_off = t * D + offs
        v_off = t * D + offs
        w_off = t * D + offs
        u_off = t * D + offs

        q_val = tl.load(Q_PTR + q_off, mask=mask)
        k_val = tl.load(K_PTR + k_off, mask=mask)
        v_val = tl.load(V_PTR + v_off, mask=mask)
        w_val = tl.load(W_PTR + w_off, mask=mask)
        u_val = tl.load(U_PTR + u_off, mask=mask)

        # Example fused RWKV-6 compute
        # hidden_state = (hidden_state + k_val * SCALE) * w_val + u_val
        hidden_state = hidden_state + (k_val * SCALE)
        hidden_state = hidden_state * w_val + u_val

        # Write output
        out_off = t * D + offs
        tl.store(OUT_PTR + out_off, hidden_state * q_val, mask=mask)

    # Store final hidden state if needed
    if STORE_FINAL_STATE != 0:
        tl.store(FINAL_H_PTR + offs, hidden_state, mask=mask)


class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, u, scale=1.0, hidden=None, return_final_state=False, reverse=False):
        USE_INITIAL_STATE = 1 if (hidden is not None) else 0
        STORE_FINAL_STATE = 1 if return_final_state else 0
        REVERSE = 1 if reverse else 0

        T, D = q.shape
        if hidden is None:
            hidden = torch.zeros(D, device=q.device, dtype=q.dtype)

        out = torch.empty_like(q)
        final_h = hidden.clone()

        # Grid setup
        BLOCK_SIZE = 128
        grid = lambda META: ( (D + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'], )

        # Launch kernel
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u,
            out, final_h,
            T, D,
            USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE,
            scale,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(q, k, v, w, u, torch.tensor(scale), hidden)
        ctx.return_final_state = return_final_state
        ctx.reverse = reverse

        if return_final_state:
            return out, final_h
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Placeholder for backward pass
        # In real usage, implement proper gradients.
        if ctx.return_final_state:
            grad_out, grad_hidden = grad_outputs
        else:
            grad_out = grad_outputs[0]
            grad_hidden = None
        q, k, v, w, u, scale, hidden = ctx.saved_tensors
        return (None, None, None, None, None, None, None, None, None)


def fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, hidden=None, return_final_state=False, reverse=False):
    return FusedRecurrentRWKV6Function.apply(q, k, v, w, u, scale, hidden, return_final_state, reverse)
