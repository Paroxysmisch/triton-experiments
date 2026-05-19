import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
    stride_logic_b, stride_logic_h, stride_logic_n,
    stride_v_b, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2) * BLOCK_N

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_logic_index = tl.load(B_Loc + cur_batch * BLOCK_N + tl.arange(0, BLOCK_N))

    v_loc_off = cur_batch_logic_index[:, None] * stride_logic_n + \
                (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_logic_n + \
                cur_head * stride_logic_h + \
                offs_d * stride_logic_b

    v_data_off = tl.arange(0, BLOCK_N)[None, :] * stride_v_b + \
                 cur_head * stride_v_h + \
                 offs_d[None, :] * stride_v_d

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_mark in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = start_mark
        start_mark = tl.multiple_of(start_mark, BLOCK_N)
        logic = tl.load(Logics + v_loc_off + start_mark * stride_logic_n, mask=(start_mark + tl.arange(0, BLOCK_N)) < cur_batch_seq_len, other=0.0)
        v = tl.load(V + v_data_off + start_mark * stride_v_b, mask=(start_mark + tl.arange(0, BLOCK_N)) < cur_batch_seq_len, other=0.0)

        cur_e_max = tl.max(tl.where((start_mark + tl.arange(0, BLOCK_N)) < cur_batch_seq_len, logic, -float("inf")), axis=0)
        e_max = tl.maximum(cur_e_max, e_max)

        logic = logic - e_max
        p = tl.exp(logic)
        acc = acc * tl.exp(e_max - cur_e_max)

        e_max = cur_e_max
        cur_batch_in_all_index = cur_batch_in_all_start_index + start_mark + tl.arange(0, BLOCK_N)
        out_off = cur_batch_in_all_index[None, :] * stride_out_b + cur_head * stride_out_h + offs_d[:, None] * stride_out_d
        acc += tl.sum(p[:, None] * v, axis=0)
        tl.store(Out + out_off, acc, mask=cur_batch_in_all_index[None, :] < cur_batch_seq_len)

    tl.debug_barrier()

    cur_batch_logic_index = tl.load(B_Loc + cur_batch * BLOCK_N + tl.arange(0, BLOCK_N))
    v_loc_off = cur_batch_logic_index[:, None] * stride_logic_n + \
                (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_logic_n + \
                cur_head * stride_logic_h + \
                offs_d * stride_logic_b

    v_data_off = tl.arange(0, BLOCK_N)[None, :] * stride_v_b + \
                 cur_head * stride_v_h + \
                 offs_d[None, :] * stride_v_d

    for start_mark in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = start_mark
        start_mark = tl.multiple_of(start_mark, BLOCK_N)
        logic = tl.load(Logics + v_loc_off + start_mark * stride_logic_n, mask=(start_mark + tl.arange(0, BLOCK_N)) < cur_batch_seq_len, other=0.0)
        v = tl.load(V + v_data_off + start_mark * stride_v_b, mask=(start_mark + tl.arange(0, BLOCK_N)) < cur_batch_seq_len, other=0.0)

        cur_batch_in_all_index = cur_batch_in_all_start_index + start_mark + tl.arange(0, BLOCK_N)
        out_off = cur_batch_in_all_index[None, :] * stride_out_b + cur_head * stride_out_h + offs_d[:, None] * stride_out_d

        logic = logic - e_max
        p = tl.exp(logic)
        acc = acc * tl.exp(e_max - logic)
        acc += tl.sum(p[:, None] * v, axis=0)

        tl.store(Out + out_off, acc, mask=cur_batch_in_all_index[None, :] < cur_batch_seq_len)


def token_softmax_reducev_fwd(logics, v, B_Loc, B_Start_Loc, B_Seqlen):
    BLOCK = 128
    batch, head_num = B_Loc.shape[0], v.shape[1]
    grid = (batch, head_num, triton.cdiv(v.shape[2], BLOCK))

    BLOCK_DMODEL = v.shape[2]
    num_warps = 4
    num_stages = 2
    assert BLOCK_DMODEL >= 128, 'BLOCK_DMODEL must greater equal 128'
    assert BLOCK_DMODEL <= 2048, 'BLOCK_DMODEL must less equal 2048'
    assert BLOCK_N == 128 or BLOCK_N == 64, 'BLOCK_N must be 64 or 128'
    return _fwd_kernel[grid](
        logics, v, v, B_Loc, B_Start_Loc, B_Seqlen,
        logics.stride(0), logics.stride(1), logics.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
