import triton
import triton.language as tl
import torch
import torch.nn.functional as F

# Define the Triton kernel
@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q,      # Input query tensor
    k,      # Input key tensor
    v,      # Input value tensor
    w,      # Input weight tensor
    u,      # Input update tensor
    b_w,    # Bias for weight
    b_u,    # Bias for update
    b_o,    # Output tensor
    hidden_state, # Hidden state tensor (optional)
    T,           # Sequence length
    D,           # Dimension of hidden state
    USE_INITIAL_STATE, # Flag to use initial hidden state
    STORE_FINAL_STATE, # Flag to store final hidden state
    REVERSE      # Flag to reverse the sequence
):
    # Get the block and grid indices
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    t = bid * triton.config.default_block_size.x + tid

    # Initialize hidden state
    h = tl.zeros((D,), dtype=tl.float32)
    if USE_INITIAL_STATE:
        h[0] = hidden_state[bid * D]

    # Iterate over the sequence length
    for i in range(T):
        idx = T - 1 - i if REVERSE else i

        # Load slices of k, v
        k_slice = tl.load(k + idx * D)
        v_slice = tl.load(v + idx * D)

        # Perform recurrent updates
        w_dot_k = tl.dot(w, k_slice)
        u_dot_k = tl.dot(u, k_slice)

        # Update hidden state
        h[0] += (1.0 - h[0]) * (w_dot_k + b_w[0]) + h[0] * (u_dot_k + b_u[0])

        # Compute output
        tl.store(b_o + idx * D, h[0] * v_slice[0])

    # Store the final hidden state if required
    if STORE_FINAL_STATE:
        hidden_state[bid * D] = h[0]

# Define the autograd function
class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, u, b_w, b_u, scale, use_initial_state, store_final_state, reverse):
        # Get sequence length and dimension
        T, D = q.shape

        # Allocate output tensor
        b_o = torch.empty_like(q)

        # Allocate hidden state tensor if required
        hidden_state = torch.empty(q.shape[0], D, device=q.device, dtype=q.dtype) if store_final_state else None

        # Configure Triton kernel
        block_size = (128, 1)
        grid_size = ((T + block_size[0] - 1) // block_size[0], q.shape[0])

        # Launch Triton kernel
        fused_recurrent_rwkv6_fwd_kernel[grid_size, block_size](
            q, k, v, w, u, b_w, b_u, b_o, hidden_state, T, D, use_initial_state, store_final_state, reverse
        )

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, w, u, b_w, b_u, b_o, hidden_state)
        ctx.scale = scale
        ctx.use_initial_state = use_initial_state
        ctx.store_final_state = store_final_state
        ctx.reverse = reverse

        return b_o, hidden_state if store_final_state else None

    @staticmethod
    def backward(ctx, grad_b_o, grad_hidden_state):
        q, k, v, w, u, b_w, b_u, b_o, hidden_state = ctx.saved_tensors
        scale = ctx.scale
        use_initial_state = ctx.use_initial_state
        store_final_state = ctx.store_final_state
        reverse = ctx.reverse

        # Get sequence length and dimension
        T, D = q.shape

        # Allocate gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_w = torch.zeros_like(w)
        grad_u = torch.zeros_like(u)
        grad_b_w = torch.zeros_like(b_w)
        grad_b_u = torch.zeros_like(b_u)

        # Configure Triton kernel for backward pass
        block_size = (128, 1)
        grid_size = ((T + block_size[0] - 1) // block_size[0], q.shape[0])

        # Launch Triton kernel for backward pass
        fused_recurrent_rwkv6_bwd_kernel[grid_size, block_size](
            grad_b_o, grad_q, grad_k, grad_v, grad_w, grad_u, grad_b_w, grad_b_u, q, k, v, w, u, b_w, b_u, hidden_state, T, D, use_initial_state, store_final_state, reverse
        )

        return grad_q, grad_k, grad_v, grad_w, grad_u, grad_b_w, grad_b_u, None, None, None, None

# Define the user-facing API
def fused_recurrent_rwkv6(q, k, v, w, u, b_w, b_u, scale=1.0, use_initial_state=False, store_final_state=False, reverse=False):
    # Scale q if required
    q_scaled = q * scale

    # Call the autograd function
    output, final_hidden_state = FusedRecurrentRWKV6Function.apply(q_scaled, k, v, w, u, b_w, b_u, scale, use_initial_state, store_final_state, reverse)

    return output, final_hidden_state if store_final_state else None
