import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=2),
    ],
    key=['NT', 'D']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, h_ptr,
    initial_state_ptr, final_state_ptr,
    B, N, D, NT, **meta
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']

    # Calculate the starting index for this block
    m_start = pid * BLOCK_SIZE_M
    n_start = bid * BLOCK_SIZE_N

    # Create block pointers for the input and output matrices
    k_block_ptr = tl.make_block_ptr(k_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))
    v_block_ptr = tl.make_block_ptr(v_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))
    d_block_ptr = tl.make_block_ptr(d_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))
    v_new_block_ptr = tl.make_block_ptr(v_new_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))
    h_block_ptr = tl.make_block_ptr(h_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))

    # Optionally handle initial and final states
    if initial_state_ptr is not None:
        initial_state_block_ptr = tl.make_block_ptr(initial_state_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))
    if final_state_ptr is not None:
        final_state_block_ptr = tl.make_block_ptr(final_state_ptr, shape=(B, N, D), strides=(N * D, D, 1), offsets=(0, m_start, 0))

    # Iterate over the time dimension
    for t in range(NT):
        # Load the necessary blocks
        k_block = tl.load(k_block_ptr)
        v_block = tl.load(v_block_ptr)
        d_block = tl.load(d_block_ptr)

        # Perform the desired computation
        result = tl.dot(k_block, v_block) + d_block

        # Store the result
        tl.store(v_new_block_ptr, result)

        # Optionally store intermediate states
        if initial_state_ptr is not None:
            tl.store(initial_state_block_ptr, result)
        if final_state_ptr is not None:
            tl.store(final_state_block_ptr, result)

        # Move to the next block
        k_block_ptr = tl.advance(k_block_ptr, 1)
        v_block_ptr = tl.advance(v_block_ptr, 1)
        d_block_ptr = tl.advance(d_block_ptr, 1)
        v_new_block_ptr = tl.advance(v_new_block_ptr, 1)
        h_block_ptr = tl.advance(h_block_ptr, 1)
        if initial_state_ptr is not None:
            initial_state_block_ptr = tl.advance(initial_state_block_ptr, 1)
        if final_state_ptr is not None:
            final_state_block_ptr = tl.advance(final_state_block_ptr, 1)


def chunk_fwd_h_fn(k, v, d, NT, initial_state=None, final_state=None):
    B, N, D = k.shape

    # Prepare output tensors
    v_new = torch.empty_like(v)
    h = torch.empty_like(v)

    # Define grid and block sizes
    grid = (triton.cdiv(N, 128), triton.cdiv(NT, 128))

    # Launch the kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, h,
        initial_state, final_state,
        B, N, D, NT
    )

    return v_new, h

# Example usage
B, N, D, NT = 32, 128, 64, 10
k = torch.randn((B, N, D), device='cuda')
v = torch.randn((B, N, D), device='cuda')
d = torch.randn((B, N, D), device='cuda')

v_new, h = chunk_fwd_h_fn(k, v, d, NT)
