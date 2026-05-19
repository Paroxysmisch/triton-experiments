import torch
import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, g, h, do, dq, dk, dg,
    B, H, T, D,
    scale: tl.constexpr,
    num_warps: tl.constexpr
):
    # Define program IDs for parallel execution
    i_bh = tl.program_id(0)
    i_h = i_bh % H

    # Compute offsets for batch and head
    batch_offset = (i_bh // H) * T * D
    head_offset = i_h * T * D

    # Define pointers for the input tensors
    p_q = tl.make_block_ptr(q + batch_offset + head_offset, (T, D), (D, 1))
    p_k = tl.make_block_ptr(k + batch_offset + head_offset, (T, D), (D, 1))
    p_v = tl.make_block_ptr(v + batch_offset + head_offset, (T, D), (D, 1))
    p_g = tl.make_block_ptr(g + batch_offset + head_offset, (T, D), (D, 1))
    p_h = tl.make_block_ptr(h + batch_offset + head_offset, (T, D), (D, 1))
    p_do = tl.make_block_ptr(do + batch_offset + head_offset, (T, D), (D, 1))

    # Define pointers for the output tensors
    p_dq = tl.make_block_ptr(dq + batch_offset + head_offset, (T, D), (D, 1))
    p_dk = tl.make_block_ptr(dk + batch_offset + head_offset, (T, D), (D, 1))
    p_dg = tl.make_block_ptr(dg + batch_offset + head_offset, (T, D), (D, 1))

    # Initialize temporary variables for intermediate results
    b_dq = tl.zeros([D], dtype=tl.float32)
    b_dk = tl.zeros([D], dtype=tl.float32)
    b_dg = tl.zeros([D], dtype=tl.float32)

    # Iterate over sequence length
    for i in range(0, T):
        # Load slices of the tensors
        b_q = tl.load(p_q + i * D)
        b_k = tl.load(p_k + i * D)
        b_v = tl.load(p_v + i * D)
        b_g = tl.load(p_g + i * D)
        b_h = tl.load(p_h + i * D)
        b_do = tl.load(p_do + i * D)

        # Compute intermediate gradients
        b_dq += tl.dot(b_do, b_k) * scale
        b_dk += tl.dot(b_do, b_q) * scale
        b_dg += tl.dot(b_do, b_h) * scale

    # Store the computed gradients
    tl.store(p_dq, b_dq)
    tl.store(p_dk, b_dk)
    tl.store(p_dg, b_dg)

def chunk_bwd_dqkg_fn(q, k, v, g, h, do, B, H, T, D, scale):
    # Prepare output tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)

    # Define grid dimensions
    grid = (B * H,)

    # Launch Triton kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, g, h, do, dq, dk, dg,
        B, H, T, D,
        scale=scale,
        num_warps=4  # Adjust number of warps based on hardware and problem size
    )

    return dq, dk, dg
