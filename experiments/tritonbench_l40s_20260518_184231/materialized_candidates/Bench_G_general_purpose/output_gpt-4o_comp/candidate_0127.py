import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(K, Dest_loc, Out, Out_scale, BLOCK_HEAD: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Get the block indices
    block_head_idx = tl.program_id(0)
    block_dmodel_idx = tl.program_id(1)
    
    # Compute the start indices for the blocks
    start_head = block_head_idx * BLOCK_HEAD
    start_dmodel = block_dmodel_idx * BLOCK_DMODEL
    
    # Load the block of data
    K_block = tl.load(K + start_head * K.stride(0) + start_dmodel, mask=True)
    
    # Compute the maximum absolute value for quantization
    max_abs_val = tl.max(tl.abs(K_block), axis=0)
    
    # Compute scale
    scale = 127.0 / max_abs_val
    tl.store(Out_scale + block_head_idx, scale)
    
    # Quantize the block
    quantized_block = tl.cast(K_block * scale, tl.int8)
    
    # Get the destination index
    dest_index = tl.load(Dest_loc + block_head_idx, mask=True)
    
    # Store the quantized data to the output
    tl.store(Out + dest_index * Out.stride(0) + start_dmodel, quantized_block, mask=True)


import torch

def destindex_copy_quantize_kv(K, Dest_loc, BLOCK_HEAD=128, BLOCK_DMODEL=128):
    # Ensure the input is on the correct device
    assert K.is_cuda and Dest_loc.is_cuda, "Input tensors must be on CUDA device"
    
    # Prepare output tensors
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((K.size(0),), dtype=torch.float32, device=K.device)
    
    # Determine grid size
    grid = (triton.cdiv(K.size(0), BLOCK_HEAD), triton.cdiv(K.size(1), BLOCK_DMODEL))
    
    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, Dest_loc, Out, Out_scale,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return Out, Out_scale
