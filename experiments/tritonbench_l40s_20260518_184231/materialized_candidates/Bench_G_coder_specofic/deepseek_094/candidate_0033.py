def destindex_copy_quantize_kv(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    k: torch.Tensor,
    dest_loc: torch.Tensor,
    head_num: int,
    block_dmodel: int,
    block_head: int,
    stream: torch.cuda.Stream):

    assert out.is_cuda and out.is_contiguous()
    assert out_scale.is_cuda and out_scale.is_contiguous()
    assert k.is_cuda and k.is_contiguous()
    assert dest_loc.is_cuda and dest_loc.is_contiguous()

    assert out.shape[0] == out_scale.shape[0]
    assert k.shape[0] == dest_loc.shape[0]

    out_stride = out.shape[1]
    scale_stride = out_scale.shape[1]

    seq_len = k.shape[1]
    total_seq_len = k.shape[0]

    grid = (head_num * block_head, )
    block = (block_dmodel, )

    _fwd_kernel_destindex_copy_quantize_kv[grid, block, stream](
        out.contiguous().view(-1).int().data_ptr(),
        out_scale.contiguous().view(-1).data_ptr(),
        k.contiguous().view(-1).int().data_ptr(),
        dest_loc.contiguous().view(-1).int().data_ptr(),
        head_num,
        block_dmodel,
        block_head,
        seq_len,
        total_seq_len,
        out_stride,
        scale_stride,
        stream.cuda_stream
    )
