import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Out_ptr, Out_scale_ptr, DestLoc_ptr,
    stride_k_batch, stride_k_head, stride_k_seq,
    stride_out_batch, stride_out_head, stride_out_seq,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    head_num: tl.constexpr,
    seq_len: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load destination location for current sequence
    dest_idx = tl.load(DestLoc_ptr + pid)
    
    # Process BLOCK_HEAD heads at a time
    for head_start in range(0, head_num, BLOCK_HEAD):
        # Calculate number of heads to process in this iteration
        head_block = tl.minimum(BLOCK_HEAD, head_num - head_start)
        
        # Initialize max values for scaling
        max_vals = tl.zeros([BLOCK_HEAD], dtype=tl.float32)
        
        # Load and find max values
        for d in range(0, BLOCK_DMODEL):
            # Load input values
            k_vals = tl.load(
                K_ptr + batch_idx * stride_k_batch +
                (head_start + tl.arange(0, head_block)) * stride_k_head +
                seq_idx * stride_k_seq + d
            )
            
            # Update max values
            max_vals = tl.maximum(max_vals, tl.abs(k_vals))
        
        # Calculate scaling factors (127.0 for int8 quantization)
        scale = max_vals / 127.0
        
        # Store scaling factors
        tl.store(
            Out_scale_ptr + batch_idx * head_num + head_start + tl.arange(0, head_block),
            scale
        )
        
        # Quantize and store values
        for d in range(0, BLOCK_DMODEL):
            k_vals = tl.load(
                K_ptr + batch_idx * stride_k_batch +
                (head_start + tl.arange(0, head_block)) * stride_k_head +
                seq_idx * stride_k_seq + d
            )
            
            # Quantize to int8
            quantized = tl.cast(tl.round(k_vals / scale), tl.int8)
            
            # Store quantized values
            tl.store(
                Out_ptr + batch_idx * stride_out_batch +
                (head_start + tl.arange(0, head_block)) * stride_out_head +
                dest_idx * stride_out_seq + d,
                quantized
            )

def destindex_copy_quantize_kv(K: torch.Tensor, DestLoc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for the Triton kernel that performs quantization of K/V tensors.
    
    Args:
        K: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
        DestLoc: Destination indices tensor of shape [batch_size * seq_len]
        
    Returns:
        tuple: (Quantized output tensor, Scale factors tensor)
    """
    batch_size, num_heads, seq_len, head_dim = K.shape
    
    # Create output tensors
    Out = torch.empty_like(K, dtype=torch.int8, device=K.device)
    Out_scale = torch.empty((batch_size, num_heads), dtype=torch.float32, device=K.device)
    
    # Calculate strides
    stride_k_batch = K.stride(0)
    stride_k_head = K.stride(1)
    stride_k_seq = K.stride(2)
    stride_out_batch = Out.stride(0)
    stride_out_head = Out.stride(1)
    stride_out_seq = Out.stride(2)
    
    # Define block sizes
    BLOCK_DMODEL = 32
    BLOCK_HEAD = 8
    
    # Launch kernel
    grid = (batch_size * seq_len,)
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, Out, Out_scale, DestLoc,
        stride_k_batch, stride_k_head, stride_k_seq,
        stride_out_batch, stride_out_head, stride_out_seq,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        head_num=num_heads,
        seq_len=seq_len,
    )
    
    return Out, Out_scale
