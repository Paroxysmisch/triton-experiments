@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, DestLoc_ptr, Out_ptr, OutScale_ptr,
    stride_k_h, stride_k_s, stride_k_d,
    stride_out_h, stride_out_s, stride_out_d,
    stride_scale_h, stride_scale_s,
    head_num, seqlen, dmodel,
    BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(0)
    # Boundary check on sequence dimension
    if pid >= seqlen:
        return

    # Load destination location
    dest_idx = tl.load(DestLoc_ptr + pid)
    # For each head
    for h in range(head_num):
        # Loop over dmodel in steps of BLOCK_DMODEL
        absmax = tl.zeros([1], dtype=tl.float32)
        d_ptr = 0
        while d_ptr < dmodel:
            offs = tl.arange(0, BLOCK_DMODEL)
            d_offs = d_ptr + offs
            mask = d_offs < dmodel

            k_offset = h * stride_k_h + pid * stride_k_s + d_offs * stride_k_d
            k_data = tl.load(K_ptr + k_offset, mask=mask, other=0.0)
            k_abs = tl.abs(k_data)
            block_absmax = tl.max(k_abs, axis=0)
            absmax = tl.maximum(absmax, block_absmax)

            d_ptr += BLOCK_DMODEL

        # Avoid division by zero
        absmax = tl.where(absmax == 0.0, 1e-8, absmax)
        scale = 127.0 / absmax
        # Store scale
        out_scale_off = h * stride_scale_h + pid * stride_scale_s
        tl.store(OutScale_ptr + out_scale_off, absmax)

        # Second pass to quantize
        d_ptr = 0
        while d_ptr < dmodel:
            offs = tl.arange(0, BLOCK_DMODEL)
            d_offs = d_ptr + offs
            mask = d_offs < dmodel

            k_offset = h * stride_k_h + pid * stride_k_s + d_offs * stride_k_d
            k_data = tl.load(K_ptr + k_offset, mask=mask, other=0.0)

            q_data = k_data * scale
            q_data = tl.round(q_data)
            q_data_int8 = tl.cast(q_data, tl.int8)

            out_offset = h * stride_out_h + dest_idx * stride_out_s + d_offs * stride_out_d
            tl.store(Out_ptr + out_offset, q_data_int8, mask=mask)

            d_ptr += BLOCK_DMODEL


def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, head_num: int):
    import triton
    import triton.language as tl

    seqlen = DestLoc.shape[0]
    dmodel = K.shape[2]
    # Assume K, Out, Out_scale are shaped [head_num, seqlen, dmodel]
    # Strides
    stride_k_h = K.stride(0)
    stride_k_s = K.stride(1)
    stride_k_d = K.stride(2)

    stride_out_h = Out.stride(0)
    stride_out_s = Out.stride(1)
    stride_out_d = Out.stride(2)

    stride_scale_h = Out_scale.stride(0)
    stride_scale_s = Out_scale.stride(1)

    BLOCK_DMODEL = 32  # one warp = 32 threads
    grid = (seqlen,)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        stride_k_h, stride_k_s, stride_k_d,
        stride_out_h, stride_out_s, stride_out_d,
        stride_scale_h, stride_scale_s,
        head_num, seqlen, dmodel,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
