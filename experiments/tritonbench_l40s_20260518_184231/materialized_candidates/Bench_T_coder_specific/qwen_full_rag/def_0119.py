import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 16}),
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 32}),
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 64}),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 16}),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 32}),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 64}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 16}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 32}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}),
    ],
    key=["in_feat", "out_feat", "ker"],
)
@triton.jit
def _pixel_shuffle_gather_jit_func(
    input, weight, bias, out, in_feat, out_feat, ker, scale
):
    """_summary_

    Args:
        input (_type_): [B, in_h, in_w, in_feat]
        weight (_type_): [out_feat, in_feat, ker, ker]
        bias (_type_): _description_
        out (_type_): _description_
        in_feat (_type_): _description_
        out_feat (_type_): _description_
        ker (_type_): _description_
        scale (_type_): _description_
    """
    # get program id
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    # do
    do = pid_n % scale
    pid_n //= scale
    # hi
    hi = pid_n % scale
    pid_n //= scale
    # wi
    wi = pid_n
    # move to center
    do = do - (scale - 1) // 2
    hi = hi - (scale - 1) // 2
    wi = wi - (scale - 1) // 2
    # offset
    n_offset = wi + hi * scale + do * scale * scale
    c_offset = tl.arange(0, BLOCK_IN_FEAT_SIZE)[:, None]
    n_offset = n_offset + c_offset
    # gather
    out_data = tl.zeros((BLOCK_OUT_FEAT_SIZE, BLOCK_KER_SIZE, BLOCK_KER_SIZE),
                        dtype=tl.float32)
    for i in range(tl.cdiv(in_feat, BLOCK_IN_FEAT_SIZE)):
        in_data = tl.load(
            input + n_offset * in_feat + i * BLOCK_IN_FEAT_SIZE,
            mask=n_offset * in_feat + i * BLOCK_IN_FEAT_SIZE <
            n_offset * in_feat + in_feat,
            other=0.0)
        ker_data = tl.load(weight + c_offset +
                           (i * BLOCK_IN_FEAT_SIZE + c_offset) * ker * ker,
                           mask=(i * BLOCK_IN_FEAT_SIZE + c_offset) *
                           ker * ker <
                           (i * BLOCK_IN_FEAT_SIZE + BLOCK_IN_FEAT_SIZE) * ker *
                           ker,
                           other=0.0)
        out_data += tl.dot(in_data, ker_data)
    out_data = out_data.to(bias.dtype)
    if bias is not None:
        out_data += bias[None, :]
    out_data = out_data.reshape((BLOCK_OUT_FEAT_SIZE, BLOCK_KER_SIZE**2))
    # store
    m = tl.arange(0, BLOCK_KER_SIZE)[None, :]
    n = tl.arange(0, BLOCK_KER_SIZE)[:, None]
    p = m * scale + n
    o_offset = (n_offset * out_feat + c_offset) * scale * scale
    for i in range(0, tl.cdiv(scale, BLOCK_SCALE_SIZE)):
        q = p + i * BLOCK_SCALE_SIZE
        mask = (o_offset + q) < o_offset + scale * scale
        tl.store(out + o_offset + q,
                 out_data[:, i * BLOCK_SCALE_SIZE:(i + 1) * BLOCK_SCALE_SIZE],
                 mask=mask)


def pixel_shuffle_gather(input, weight, bias, scale):
    """
    Args:
        input (Tensor): (batch, in_h, in_w, in_feat)
        weight (Tensor): (out_feat, in_feat, kernel_size, kernel_size)
        bias (Tensor): (out_feat,)
        scale (int): scale factor
    Return:
        output (Tensor): (batch, out_h, out_w, out_feat)
                         out_h = in_h * scale
                         out_w = in_w * scale
                         out_feat = out_feat
    """
    batch, in_h, in_w, in_feat = input.shape
    out_feat, _, ker, _ = weight.shape
    assert ker % 2 != 0, "kernel size must be odd"
    assert in_feat % scale == 0, "in_feat must divisible by scale"
    out_h, out_w = in_h * scale, in_w * scale
    out = torch.empty(batch, out_h, out_w, out_feat,
                      dtype=input.dtype, device="cuda")
    assert (
        in_feat <= 128
    ), "inner feature should less than or equal to 128, use groupwise instead"
    BLOCK_IN_FEAT_SIZE = max(128 // scale, 1)
    BLOCK_OUT_FEAT_SIZE = min(128, out_feat)
    BLOCK_SCALE_SIZE = 16
    BLOCK_KER_SIZE = ker
    num_warps = 4
    num_stages = 2
    grid = (batch * (in_h - ker + 1) // scale,
            out_feat // BLOCK_OUT_FEAT_SIZE)
    _pixel_shuffle_gather_jit_func[grid](input, weight, bias, out, in_feat,
                                         out_feat, ker, scale, num_warps=num_warps,
                                         num_stages=num_stages,
                                         BLOCK_IN_FEAT_SIZE=BLOCK_IN_FEAT_SIZE,
                                         BLOCK_OUT_FEAT_SIZE=BLOCK_OUT_FEAT_SIZE,
                                         BLOCK_SCALE_SIZE=BLOCK_SCALE_SIZE,
                                         BLOCK_KER_SIZE=BLOCK_KER_SIZE)
    return out
