@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr,
    Dest_loc_ptr,
    Out_ptr,
    Out_scale_ptr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    seq_len: tl.constexpr,
    n_head: tl.constexpr,
    n_kv: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    OUT_SCALE_DTYPE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    num_iters: tl.constexpr,
    **kwargs
):
    pass
