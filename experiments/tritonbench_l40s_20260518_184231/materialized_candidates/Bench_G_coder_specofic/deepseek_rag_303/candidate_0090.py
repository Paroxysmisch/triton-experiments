import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["V", "Out", "B_Loc", "B_Start_Loc", "B_Seqlen"],
)
@triton.jit
def _fwd_kernel(
    V,
    Out,
    B_Loc,
    B_Start_Loc,
    B_Seqlen,
    stride_v_batch,
    stride_v_head,
    stride_v_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    N_HEADS,
    D_MODEL,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id_1 = tl.program_id(1)
    program_id_0 = tl.program_id(0)
    start_n = program_id_1 * BLOCK_N
    V += program_id_0 * stride_v_batch + program_id_1 * stride_v_head
    Out += program_id_0 * stride_out_batch + program_id_1 * stride_out_head
    col_offsets_d = tl.arange(0, BLOCK_DMODEL)
    V += col_offsets_d
    Out += col_offsets_d
    B_Loc += program_id_0
    B_Start_Loc += program_id_0
    stride_v_batch = 0
    stride_v_head = 0
    stride_out_batch = 0
    stride_out_head = 0
    col_offsets_d = 0
    B_Seqlen += program_id_0
    B_N_Loc = tl.load(B_Loc)
    B_N_Seqlen = tl.load(B_Seqlen)
    start_n = min(start_n, B_N_Loc)
    B_N_Start_Loc = tl.load(B_Start_Loc)
    if B_N_Loc == 0:
        return
    block_n_count = start_n + BLOCK_N - 1
    block_n_count = tl.minimum(block_n_count, B_N_Loc - 1)
    for a in range(start_n, block_n_count + 1, 1):
        a_v_0 = tl.load(V + a * stride_v_batch)
        a_v_1 = tl.load(V + a * stride_v_batch + stride_v_dim)
        mask = col_offsets_d < D_MODEL
        tl.device_assert(
            0
            != tl.where(
                mask,
                a_v_0 != float("-inf"),  # noqa: E711
                False,
            ),
            "none of index in mask",
        )
        e_max = tl.zeros([2], dtype=tl.float32)
        acc = tl.zeros([2], dtype=tl.float32)
        index_0 = a * stride_v_batch
        index_1 = a * stride_v_batch + stride_v_dim
        seq_loc = B_N_Start_Loc + a
        v_prev = tl.zeros([1, 2], dtype=tl.float32)
        v_prev[0, 1] = float("-inf")
        for i in range(
            tl.load(B_Start_Loc + seq_loc) >> LOG_PROGRAM_N,
            (tl.load(B_Start_Loc + seq_loc + 1) + PROGRAM_N - 1) >> LOG_PROGRAM_N,
            1,
        ):
            start_idx = i << LOG_PROGRAM_N
            v = tl.load(V + index_0 + start_idx, mask=(col_offsets_d + start_idx) < N_DIM)
            v += tl.load(
                V + index_1 + start_idx, mask=(col_offsets_d + start_idx) < N_DIM
            )
            v_prev_exp = tl.exp(v_prev - a_v_0)
            p = v_prev_exp * a_v_1 + tl.exp(v - a_v_0) * tl.broadcast_to(a_v_1, 2)
            e_max_ = tl.maximum(e_max, v)
            coef = tl.exp(e_max - e_max_)
            acc *= coef
            acc += p
            e_max = e_max_
            v_prev = v
        v_exp = tl.exp(v_prev - a_v_0)
        p = v_exp * a_v_1
        scale = tl.exp(e_max - a_v_0)
        acc *= scale
        acc += p * tl.broadcast_to(a_v_1, 2)
        acc = acc / (tl.exp(e_max - B_N_Loc) - 1)
        acc_0 = acc[0]
        acc_1 = acc[1]
        tl.store(Out + a * stride_out_batch, acc_0)
        tl.store(Out + a * stride_out_batch + stride_out_dim, acc_1)
    tl.debug_barrier()

def token_softmax_reducev_fwd(delta, atten, logit_bias, b_loc, b_start_loc, b_seqlen):
    BLOCK_DMODEL = 2
    odim = atten.shape
    nheads = odim[1]
    (batch,) = b_loc.shape
    batch = max(16, batch)
    for i in range(0, batch):
        _fwd_kernel[(i, 0)](
            logit_bias,
            delta,
            b_loc,
            b_start_loc,
            b_seqlen,
            logit_bias.stride(0),
            logit_bias.stride(1),
            logit_bias.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            nheads,
            logit_bias.shape[3],
            BLOCK_DMODEL=BLOCK_DMODEL,
            grid=(batch, nheads),
        )
