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
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
     
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :], 
                       mask=offs_g[:, None] < group_size, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.).to(tl.float16)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
    
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g  +  offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None]<group_size)
    tl.store(os_ptrs, data_scale)
    return
