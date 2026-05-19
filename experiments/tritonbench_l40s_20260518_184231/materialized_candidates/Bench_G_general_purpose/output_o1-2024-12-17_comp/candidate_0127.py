import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, 
    Dest_loc_ptr, 
    Out_ptr, 
    Out_scale_ptr,
    BLOCK_HEAD: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    N_HEADS: tl.constexpr, 
    N_SEQ: tl.constexpr, 
    D_MODEL: tl.constexpr
):
    head_block_id = tl.program_id(0)
    seq_id = tl.program_id(1)

    head_start = head_block_id * BLOCK_HEAD
    heads = head_start + tl.arange(0, BLOCK_HEAD)
    head_mask = heads < N_HEADS
    d_idx = tl.arange(0, BLOCK_DMODEL)

    # Load the destination location for this seq_id
    # boundary check on seq
    if seq_id >= N_SEQ:
        return
    dest_idx = tl.load(Dest_loc_ptr + seq_id, mask=seq_id < N_SEQ, other=0)

    # For each head in this block:
    for h_ofs in range(BLOCK_HEAD):
        h = head_start + h_ofs
        if h >= N_HEADS:
            break

        # Compute pointer offset for reading from K
        k_offset = (h * N_SEQ + seq_id) * D_MODEL
        # Gather data from K
        k_values = tl.load(
            K_ptr + k_offset + d_idx,
            mask=(d_idx < D_MODEL),
            other=0.0
        )

        # Find max abs value
        abs_vals = tl.abs(k_values)
        max_abs = tl.maximum(tl.max(abs_vals, axis=0), 1e-8)

        # Compute scale (reciprocal for quant)
        r_scale = 127.0 / max_abs

        # Quantize
        scaled = k_values * r_scale
        clamped = tl.maximum(tl.minimum(scaled, 127.0), -128.0)
        quantized = clamped.to(tl.int8)

        # Write quantized data
        out_offset = (h * N_SEQ + dest_idx) * D_MODEL
        tl.store(
            Out_ptr + out_offset + d_idx,
            quantized,
            mask=(d_idx < D_MODEL)
        )

        # Write scale
        scale_offset = (h * N_SEQ + dest_idx)
        tl.store(
            Out_scale_ptr + scale_offset,
            1.0 / r_scale,
            mask=head_mask
        )


@torch.no_grad()
def destindex_copy_quantize_kv(K, Dest_loc):
    B_HEAD = 1
    B_DMODEL = 128
    N_HEADS, N_SEQ, D_MODEL = K.shape
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((N_HEADS, N_SEQ), dtype=torch.float32, device=K.device)

    grid = ( (N_HEADS + B_HEAD - 1) // B_HEAD, N_SEQ )
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, 
        Dest_loc, 
        Out, 
        Out_scale,
        BLOCK_HEAD=B_HEAD,
        BLOCK_DMODEL=B_DMODEL,
        N_HEADS=N_HEADS, 
        N_SEQ=N_SEQ, 
        D_MODEL=D_MODEL
    )
    return Out, Out_scale
