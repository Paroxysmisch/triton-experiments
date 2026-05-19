import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    ProbPtr, XPtr, TRemPtr, OutPtr,
    VLen, CLen, BLk,
    DT_TYPE: tl.constexpr, VALUE_TYPE: tl.constexpr,
):
    """
    Forward kernel for Segment-Sum-pooling.
    :param ProbPtr: Pointer to the probability matrix.
    :param XPtr: Pointer to the input feature matrix.
    :param TRemPtr: Pointer to the remaining time matrix.
    :param OutPtr: Pointer to the output matrix.
    :param VLen: Length of the features.
    :param CLen: Total length of the sequences.
    :param BLk: Block size.
    :param DT_TYPE: Data type of the probability and output matrices.
    :param VALUE_TYPE: Data type of the feature and remaining time matrices.
    """
    seq_index = tl.program_id(0)
    feat_blocks = tl.program_id(1)

    prob_ptr = ProbPtr + seq_index * CLen + feat_blocks * BLk
    x_ptr = XPtr + seq_index * CLen + feat_blocks * BLk
    t_remaining_ptr = TRemPtr + seq_index * CLen + feat_blocks * BLk
    out_ptr = OutPtr + seq_index * VLen + feat_blocks * BLk

    row_offset = seq_index * CLen
    col_offsets = tl.arange(0, BLk)
    valid_col_mask = col_offsets < CLen

    p_values = tl.load(prob_ptr + col_offsets, mask=valid_col_mask, other=0.0).to(DT_TYPE)
    x_values = tl.load(x_ptr + col_offsets, mask=valid_col_mask, other=0.0).to(VALUE_TYPE)
    t_remaining_values = tl.load(t_remaining_ptr + col_offsets, mask=valid_col_mask, other=0.0).to(VALUE_TYPE)

    sum_p = tl.sum(p_values)
    normalized_p = p_values / sum_p
    weighted_x = x_values * normalized_p
    weighted_t_remaining = t_remaining_values * normalized_p

    out_value = tl.sum(weighted_x) + weighted_t_remaining

    tl.store(out_ptr + col_offsets, out_value.to(DT_TYPE), mask=col_offsets < VLen)


class _triton_ssp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prob, x, t_remaining, dt_dtype=torch.float32, value_dtype=None):
        """
        Forward pass for Segment-Sum-Pooling.
        :param ctx: Autograd context.
        :param prob: Probability matrix.
        :param x: Input feature matrix.
        :param t_remaining: Remaining time matrix.
        :param dt_dtype: Data type for probability and output matrices.
        :param value_dtype: Data type for feature and remaining time matrices.
        :return: Pooled feature matrix.
        """
        check_last_dim_equal(prob, x, "prob", "x")
        check_last_dim_equal(prob, t_remaining, "prob", "t_remaining")

        seq_len = prob.shape[-2]
        value_dim = x.shape[-1]

        value_dtype = x.dtype if value_dtype is None else value_dtype

        num_block = triton.cdiv(value_dim, BLOCK)

        out = torch.empty((prob.shape[0], num_block),
                          device=x.device,
                          dtype=value_dtype)

        grid = (prob.shape[0], num_block,)
        ctx.mark_dirty(prob, x, t_remaining)

        _fwd_kernel[grid](
            prob,
            x,
            t_remaining,
            out,
            value_dim,
            seq_len,
            BLOCK,
            dt_dtype=dt_dtype,
            value_dtype=value_dtype,
            num_warps=num_warps(),
        )

        return out
