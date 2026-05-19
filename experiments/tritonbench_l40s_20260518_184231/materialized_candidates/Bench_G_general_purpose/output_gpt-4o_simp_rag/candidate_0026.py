import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_g, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_g, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr 
):
    # Identify the current program ID for indexing
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
     
    # Define offsets for group and dimension
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    # Load destination index
    dest_index = tl.load(Dest_loc + cur_index)

    # Load source data from tensor K
    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :], 
                       mask=offs_g[:, None] < group_size, other=0.0)
    
    # Calculate absolute values and scale for quantization
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.).to(tl.float16)
    
    # Quantize the source data
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
    
    # Define output pointers
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    
    # Store quantized data and scale
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale)

@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    # Extract dimensions and define constants
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    # Ensure head_dim is divisible by quant_group_dim
    assert head_dim % quant_group_dim == 0, "Error: head_dim must be divisible by quant_group_dim"

    # Define grid and block configurations
    grid = (seq_len, head_num)
    num_warps = 1

    # Calculate group size and dimension
    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    # Reshape tensors for processing
    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim, 
        num_warps=num_warps,
        num_stages=1,
    )
