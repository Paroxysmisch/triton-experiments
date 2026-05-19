import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def fused_recurrent_rwkv6_kernel(
    r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr, final_state_ptr,
    scale, initial_state_ptr, n, reverse, output_final_state,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n)

    # Load initial state if provided
    if initial_state_ptr != 0:
        state = tl.load(initial_state_ptr)
    else:
        state = tl.zeros((1,), dtype=tl.float32)

    for i in range(block_start, block_end):
        idx = n - 1 - i if reverse else i
        r = tl.load(r_ptr + idx)
        k = tl.load(k_ptr + idx)
        v = tl.load(v_ptr + idx)
        w = tl.load(w_ptr + idx)
        u = tl.load(u_ptr + idx)

        # Compute the fused recurrent operation
        state = state * w + r * tl.exp(k * scale) * v
        output = state + u

        # Store the result
        tl.store(o_ptr + idx, output)

    # Store the final state if needed
    if output_final_state:
        tl.store(final_state_ptr, state)

# Define the Python wrapper function
def fused_recurrent_rwkv6(r, k, v, w, u, scale=1.0, initial_state=None, reverse=False, output_final_state=False):
    n = r.shape[0]
    BLOCK_SIZE = 128  # Adjust this based on your GPU capabilities

    # Allocate output tensors
    o = torch.empty_like(r)
    final_state = torch.empty(1, dtype=r.dtype) if output_final_state else None

    # Prepare pointers
    r_ptr = r.data_ptr()
    k_ptr = k.data_ptr()
    v_ptr = v.data_ptr()
    w_ptr = w.data_ptr()
    u_ptr = u.data_ptr()
    o_ptr = o.data_ptr()
    initial_state_ptr = initial_state.data_ptr() if initial_state is not None else 0
    final_state_ptr = final_state.data_ptr() if final_state is not None else 0

    # Launch the Triton kernel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_recurrent_rwkv6_kernel[grid](
        r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr, final_state_ptr,
        scale, initial_state_ptr, n, reverse, output_final_state,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return (o, final_state) if output_final_state else o

# Example usage
r = torch.randn(1024, dtype=torch.float32, device='cuda')
k = torch.randn(1024, dtype=torch.float32, device='cuda')
v = torch.randn(1024, dtype=torch.float32, device='cuda')
w = torch.randn(1024, dtype=torch.float32, device='cuda')
u = torch.randn(1024, dtype=torch.float32, device='cuda')
initial_state = torch.tensor([0.0], dtype=torch.float32, device='cuda')

o, final_state = fused_recurrent_rwkv6(r, k, v, w, u, scale=0.5, initial_state=initial_state, reverse=False, output_final_state=True)
