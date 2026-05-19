import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Dest_loc, Out,                    # Tensors
    stride_k_bs, stride_k_h, stride_k_d,  # Strides for K tensor
    stride_o_bs, stride_o_h, stride_o_d,  # Strides for Out tensor
    head_num, head_dim,                   # Dimensions
    BLOCK_DMODEL: tl.constexpr,          # Block sizes
    BLOCK_HEAD: tl.constexpr,
):
    # Get current program instance ID
    cur_index = tl.program_id(0)
    
    # Create offset arrays for heads and dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load destination index
    dest_index = tl.load(Dest_loc + cur_index)
    
    # Calculate pointers for input and output
    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    
    # Create mask for valid elements
    mask = (offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim)
    
    # Load and store data
    k = tl.load(k_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, k, mask=mask)

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,         # Tensors
    stride_k_bs, stride_k_h, stride_k_d,  # Strides for K tensor
    stride_o_bs, stride_o_h, stride_o_d,  # Strides for Out tensor
    stride_os_bs, stride_os_h, stride_os_d,  # Strides for Out_scale tensor
    head_num, head_dim,                   # Dimensions
    BLOCK_DMODEL: tl.constexpr,          # Block sizes
    BLOCK_HEAD: tl.constexpr,
):
    # Get current program instance ID
    cur_index = tl.program_id(0)
    
    # Create offset arrays for heads and dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load destination index
    dest_index = tl.load(Dest_loc + cur_index)
    
    # Create mask for valid elements
    mask = (offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim)
    
    # Load source data
    src_ptr = K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :]
    src_data = tl.load(src_ptr, mask=mask, other=0.0)
    
    # Compute scale (max absolute value / 127.0 for int8 quantization)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)[:, None]
    
    # Quantize data to int8
    q_src_data = (src_data / data_scale).to(tl.int8)
    
    # Store quantized data and scales
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    
    tl.store(o_ptrs, q_src_data, mask=mask)
    tl.store(os_ptrs, data_scale, mask=(offs_h[:, None] < head_num))
