import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_g, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_g, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_g,
    group_size: tl.constexpr,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr
):
    # Determine processing block coordinates
    cur_index = tl.program_id(0)  # Sequence position
    cur_head = tl.program_id(1)    # Attention head
    
    # Create offset ranges for vectorized operations
    group_offsets = tl.arange(0, BLOCK_GROUP_NUM)
    dim_offsets = tl.arange(0, BLOCK_GROUP_DIM)
    
    # Load destination index from memory
    dest_idx = tl.load(Dest_loc + cur_index)
    
    # Calculate source pointer with 2D block structure
    src_ptr = K + cur_index * stride_k_bs + cur_head * stride_k_h + \
              group_offsets[:, None] * stride_k_g + dim_offsets[None, :]
    
    # Load and mask source data
    src_data = tl.load(
        src_ptr,
        mask=group_offsets[:, None] < group_size,
        other=0.0
    )
    
    # Compute quantization parameters
    abs_vals = tl.abs(src_data)
    group_max = tl.max(abs_vals, axis=1)
    scale_values = (group_max / 127.0).to(tl.float16)
    
    # Quantize data to int8 using computed scales
    quantized_data = (src_data / scale_values[:, None]).to(tl.int8)
    
    # Calculate output pointers
    out_ptr = Out + dest_idx * stride_o_bs + cur_head * stride_o_h + \
              group_offsets[:, None] * stride_o_g + dim_offsets[None, :]
    scale_ptr = Out_scale + dest_idx * stride_os_bs + cur_head * stride_os_h + group_offsets
    
    # Store results with boundary checks
    tl.store(out_ptr, quantized_data, mask=group_offsets[:, None] < group_size)
    tl.store(scale_ptr, scale_values, mask=group_offsets < group_size)

@torch.no_grad()
def destindex_copy_quantize_kv(K: torch.Tensor, DestLoc: torch.Tensor, 
                              Out: torch.Tensor, Out_scale: torch.Tensor):
    # Validate tensor dimensions
    assert K.dim() == 3, "Input K must be 3D (seq, heads, dim)"
    assert Out.dim() == 4, "Output must be 4D (seq, heads, groups, group_dim)"
    
    seq_len = DestLoc.size(0)
    num_heads = K.size(1)
    orig_dim = K.size(2)
    group_dim = 8  # Fixed group dimension for quantization
    
    # Configure kernel launch parameters
    grid = (seq_len, num_heads)
    group_size = orig_dim // group_dim
    
    # Reshape tensors for group processing
    K_view = K.view(K.size(0), num_heads, group_size, group_dim)
    Out_view = Out.view(Out.size(0), num_heads, group_size, group_dim)
    
    # Calculate optimal block sizes
    BLOCK_GROUP_NUM = triton.next_power_of_2(group_size)
    BLOCK_GROUP_DIM = group_dim
    
    # Launch kernel with optimized execution parameters
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K_view, DestLoc, Out_view, Out_scale,
        K_view.stride(0), K_view.stride(1), K_view.stride(2), K_view.stride(3),
        Out_view.stride(0), Out_view.stride(1), Out_view.stride(2), Out_view.stride(3),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=BLOCK_GROUP_NUM,
        BLOCK_GROUP_DIM=BLOCK_GROUP_DIM,
        num_warps=4 if BLOCK_GROUP_NUM >= 64 else 1,
        num_stages=3
    )
