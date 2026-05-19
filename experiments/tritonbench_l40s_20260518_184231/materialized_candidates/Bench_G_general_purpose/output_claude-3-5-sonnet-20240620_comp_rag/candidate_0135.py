import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Dest_loc, Out,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num, head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    # Get current index from program ID
    cur_index = tl.program_id(0)
    
    # Create offset arrays for heads and dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load destination index
    dest_index = tl.load(Dest_loc + cur_index)
    
    # Calculate source and destination pointers
    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    
    # Load and store with masking
    mask = (offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim)
    k = tl.load(k_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, k, mask=mask)

@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    
    # Validate tensor shapes
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    
    # Calculate block sizes
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    
    # Configure kernel launch parameters
    grid = (seq_len,)
    num_warps = 1

    # Launch kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num, head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num, head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    # Get current index from program ID
    cur_index = tl.program_id(0)
    
    # Create offset arrays for heads and dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load destination index
    dest_index = tl.load(Dest_loc + cur_index)
    
    # Load source data with masking
    mask = (offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim)
    src_data = tl.load(
        K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :],
        mask=mask,
        other=0.0,
    )
    
    # Compute quantization scale
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)[:, None]
    
    # Quantize data
    q_src_data = (src_data / data_scale).to(tl.int8)
    
    # Calculate output pointers
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    
    # Store quantized data and scales
    tl.store(o_ptrs, q_src_data, mask=mask)
    tl.store(os_ptrs, data_scale, mask=(offs_h[:, None] < head_num))

@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    
    # Validate tensor shapes
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    
    # Calculate block sizes
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    
    # Configure kernel launch parameters
    grid = (seq_len,)
    num_warps = 1

    # Launch kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        head_num, head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
