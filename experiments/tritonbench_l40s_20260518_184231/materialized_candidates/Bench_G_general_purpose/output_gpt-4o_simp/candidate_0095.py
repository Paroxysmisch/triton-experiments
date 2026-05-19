import torch
import triton
import triton.language as tl

# Forward kernel
@triton.jit
def recurrent_retention_forward_kernel(Q, K, V, O, T, BK, BV, Lq, Lk, Lv, Lo, initial_state_ptr=None, final_state_ptr=None):
    pid = tl.program_id(0)
    # Load blocks of Q, K, V
    q = tl.load(Q + pid * BK, mask=pid < Lq, other=0.0)
    k = tl.load(K + pid * BK, mask=pid < Lk, other=0.0)
    v = tl.load(V + pid * BV, mask=pid < Lv, other=0.0)

    # Scale query
    q = q * tl.sqrt(1.0 / BK)

    # Initialize state
    if initial_state_ptr is not None:
        state = tl.load(initial_state_ptr + pid * BK, mask=pid < Lq, other=0.0)
    else:
        state = tl.zeros([BK], dtype=tl.float32)

    # Iterate over time steps
    for t in range(T):
        # Compute key-value product
        h = tl.dot(k, v)
        # Update state
        state = state + h
        # Compute output
        o = tl.dot(q, state)
        tl.store(O + pid * BV + t * BV, o, mask=pid < Lo)

    # Store final state if needed
    if final_state_ptr is not None:
        tl.store(final_state_ptr + pid * BK, state, mask=pid < Lq)

# Backward kernel
@triton.jit
def recurrent_retention_backward_kernel(dO, Q, K, V, dQ, dK, dV, T, BK, BV, Lq, Lk, Lv, Lo):
    pid = tl.program_id(0)
    # Load blocks of dO, Q, K, V
    do = tl.load(dO + pid * BV, mask=pid < Lo, other=0.0)
    q = tl.load(Q + pid * BK, mask=pid < Lq, other=0.0)
    k = tl.load(K + pid * BK, mask=pid < Lk, other=0.0)
    v = tl.load(V + pid * BV, mask=pid < Lv, other=0.0)

    # Initialize gradients
    dq = tl.zeros([BK], dtype=tl.float32)
    dk = tl.zeros([BK], dtype=tl.float32)
    dv = tl.zeros([BV], dtype=tl.float32)

    # Iterate over time steps in reverse
    for t in reversed(range(T)):
        # Compute gradients
        dq += tl.dot(do, k)
        dk += tl.dot(q, do)
        dv += tl.dot(k, do)

    # Store gradients
    tl.store(dQ + pid * BK, dq, mask=pid < Lq)
    tl.store(dK + pid * BK, dk, mask=pid < Lk)
    tl.store(dV + pid * BV, dv, mask=pid < Lv)

# Python wrapper
class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None):
        # Determine block sizes and launch grid
        BK = 128  # Example block size for keys
        BV = 128  # Example block size for values
        T = q.shape[0]  # Temporal dimension
        Lq, Lk, Lv, Lo = q.shape[1], k.shape[1], v.shape[1], q.shape[1]

        # Allocate output tensors
        o = torch.empty_like(q)
        final_state = torch.empty_like(q) if initial_state is not None else None

        # Launch forward kernel
        recurrent_retention_forward_kernel[(1,)](q, k, v, o, T, BK, BV, Lq, Lk, Lv, Lo, initial_state, final_state)

        # Save for backward
        ctx.save_for_backward(q, k, v, o, initial_state)

        return (o, final_state) if final_state is not None else o

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, o, initial_state = ctx.saved_tensors
        BK = 128  # Example block size for keys
        BV = 128  # Example block size for values
        T = q.shape[0]  # Temporal dimension
        Lq, Lk, Lv, Lo = q.shape[1], k.shape[1], v.shape[1], q.shape[1]

        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Launch backward kernel
        recurrent_retention_backward_kernel[(1,)](do, q, k, v, dq, dk, dv, T, BK, BV, Lq, Lk, Lv, Lo)

        return dq, dk, dv, None

# Main function
def fused_recurrent_retention(q, k, v, initial_state=None):
    return FusedRecurrentRetentionFunction.apply(q, k, v, initial_state)
