import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Out_ptr, Out_scale_ptr, DestLoc_ptr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_HEAD: tl.constexpr, SEQ_LEN: tl.constexpr
):
    # Get program id
    pid = tl.program_id(0)

    # Calculate head and sequence index
    head_idx = pid // SEQ_LEN
    seq_idx = pid % SEQ_LEN

    # Compute the destination index
    dest_idx = tl.load(DestLoc_ptr + seq_idx)

    # Load the block of data for the current head and sequence
    K_offset = head_idx * BLOCK_HEAD * BLOCK_DMODEL + seq_idx * BLOCK_DMODEL
    K_block = tl.load(K_ptr + K_offset, mask=True)

    # Calculate the absolute maximum for scaling
    abs_max = tl.max(tl.abs(K_block))

    # Compute scaling factor
    scale = abs_max / 127.0
    inv_scale = 1.0 / scale

    # Quantize the data to int8
    K_quantized = tl.cast(tl.round(K_block * inv_scale), tl.int8)

    # Store quantized data and scale
    Out_offset = head_idx * BLOCK_HEAD * BLOCK_DMODEL + dest_idx * BLOCK_DMODEL
    tl.store(Out_ptr + Out_offset, K_quantized, mask=True)
    tl.store(Out_scale_ptr + head_idx * SEQ_LEN + dest_idx, scale)

# Python wrapper function
def destindex_copy_quantize_kv(K, DestLoc, BLOCK_DMODEL, BLOCK_HEAD):
    # Get dimensions
    head_num, seq_len, dmodel = K.shape

    # Allocate output tensors
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((head_num, seq_len), dtype=torch.float32)

    # Define grid and block sizes
    grid = (head_num * seq_len,)
    block = (BLOCK_DMODEL,)

    # Launch Triton kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](K, Out, Out_scale, DestLoc,
                                                 BLOCK_DMODEL=BLOCK_DMODEL,
                                                 BLOCK_HEAD=BLOCK_HEAD,
                                                 SEQ_LEN=seq_len)
    return Out, Out_scale

# Example usage
K = torch.randn(8, 16, 64, device='cuda')  # Example tensor
DestLoc = torch.randint(0, 16, (16,), device='cuda')  # Random destination indices
BLOCK_DMODEL = 64
BLOCK_HEAD = 8

Out, Out_scale = destindex_copy_quantize_kv(K, DestLoc, BLOCK_DMODEL, BLOCK_HEAD)
