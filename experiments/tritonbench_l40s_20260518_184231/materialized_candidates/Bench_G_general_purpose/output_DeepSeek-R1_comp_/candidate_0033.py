import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,  # Pointer to the input tensor K
    Out,  # Pointer to the output tensor Out (int8)
    Out_scale,  # Pointer to the output scale tensor
    DestLoc,  # Pointer to the destination indices
    stride_kt, stride_kh, stride_kd,  # Strides for K tensor
    stride_ot, stride_oh, stride_od,  # Strides for Out tensor
    stride_ost, stride_osh,  # Strides for Out_scale tensor
    seq_len,  # Sequence length of the input
    head_num,  # Number of heads
    dmodel,  # Model dimension
    BLOCK_DMODEL: tl.constexpr,  # Block size for dmodel dimension
    BLOCK_HEAD: tl.constexpr,  # Block size for head dimension
):
    # Determine the current sequence index and head index
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Check if the current indices are within bounds
    if pid_seq >= seq_len or pid_head >= head_num:
        return
    
    # Load the destination index for the current sequence element
    dest_s = tl.load(DestLoc + pid_seq)
    
    # Calculate base offsets for K and Out
    off_k_base = pid_seq * stride_kt + pid_head * stride_kh
    off_out_base = dest_s * stride_ot + pid_head * stride_oh
    off_scale = dest_s * stride_ost + pid_head * stride_osh
    
    max_abs = tl.zeros((1,), dtype=tl.float32)
    
    # First pass: compute max absolute value over dmodel
    for d_off in range(0, dmodel, BLOCK_DMODEL):
        d = d_off + tl.arange(0, BLOCK_DMODEL)
        mask = d < dmodel
        
        # Load K values
        k = tl.load(K + off_k_base + d * stride_kd, mask=mask, other=0.0)
        current_abs = tl.abs(k)
        current_max = tl.max(current_abs, axis=0)
        max_abs = tl.maximum(max_abs, current_max)
    
    # Compute scale factor
    scale = max_abs / 127.0
    
    # Second pass: quantize and store
    for d_off in range(0, dmodel, BLOCK_DMODEL):
        d = d_off + tl.arange(0, BLOCK_DMODEL)
        mask = d < dmodel
        
        # Load K values again for quantization
        k = tl.load(K + off_k_base + d * stride_kd, mask=mask, other=0.0)
        quantized = tl.libdevice.llrint(k / scale)
        
        # Store quantized values to Out
        tl.store(Out + off_out_base + d * stride_od, quantized, mask=mask)
    
    # Store the scale factor
    tl.store(Out_scale + off_scale, scale)

def destindex_copy_quantize_kv(K: torch.Tensor, DestLoc: torch.Tensor):
    assert K.is_cuda and DestLoc.is_cuda, "Inputs must be on CUDA device"
    seq_len = DestLoc.size(0)
    head_num, dmodel = K.shape[1], K.shape[2]
    
    # Determine the maximum destination index to size the output tensors
    max_dest = DestLoc.max().item()
    dest_seq_len = max_dest + 1  # Assuming destination indices are 0-based
    
    # Initialize output tensors
    Out = torch.empty((dest_seq_len, head_num, dmodel), dtype=torch.int8, device=K.device)
    Out_scale = torch.empty((dest_seq_len, head_num), dtype=K.dtype, device=K.device)
    
    # Define kernel grid and block dimensions
    BLOCK_DMODEL = 128  # Adjust based on dmodel for optimal performance
    BLOCK_HEAD = 1  # Each block processes one head at a time
    
    grid = (seq_len, head_num)
    
    # Launch the kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, Out, Out_scale, DestLoc,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale.stride(0), Out_scale.stride(1),
        seq_len, head_num, dmodel,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_HEAD=BLOCK_HEAD
    )
    
    return Out, Out_scale
