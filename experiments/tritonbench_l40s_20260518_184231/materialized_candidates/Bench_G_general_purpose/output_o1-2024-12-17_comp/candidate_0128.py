import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_PTR, OUT_PTR,
    COS_PTR, SIN_PTR,
    CU_SEQLENS_PTR,  # optional, can be None
    B, H, M, D,
    STRIDE_XB, STRIDE_XH, STRIDE_XM, STRIDE_XD,
    STRIDE_OB, STRIDE_OH, STRIDE_OM, STRIDE_OD,
    STRIDE_CB, STRIDE_CH, STRIDE_CM, STRIDE_CD,
    STRIDE_SB, STRIDE_SH, STRIDE_SM, STRIDE_SD,
    CONJUGATE, INTERLEAVED,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    # program_id for batch, head, M-dim
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # block-ranges for M, D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    # clamp
    mask_m = offs_m < M
    mask_d = offs_d < D

    # handle optional cu_seqlens
    if CU_SEQLENS_PTR is not None:
        seq_offset = tl.load(CU_SEQLENS_PTR + pid_batch)
        seq_idx = seq_offset + offs_m
    else:
        seq_idx = offs_m

    # base pointers
    x_ptrs = X_PTR + (pid_batch * STRIDE_XB
                      + pid_head * STRIDE_XH
                      + offs_m[:, None] * STRIDE_XM
                      + offs_d[None, :] * STRIDE_XD)
    out_ptrs = OUT_PTR + (pid_batch * STRIDE_OB
                          + pid_head * STRIDE_OH
                          + offs_m[:, None] * STRIDE_OM
                          + offs_d[None, :] * STRIDE_OD)
    cos_ptrs = COS_PTR + (pid_batch * STRIDE_CB
                          + pid_head * STRIDE_CH
                          + seq_idx[:, None] * STRIDE_CM
                          + offs_d[None, :] * STRIDE_CD)
    sin_ptrs = SIN_PTR + (pid_batch * STRIDE_SB
                          + pid_head * STRIDE_SH
                          + seq_idx[:, None] * STRIDE_SM
                          + offs_d[None, :] * STRIDE_SD)

    # load data
    x = tl.where(mask_m[:, None] & mask_d[None, :], tl.load(x_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.), 0.)
    cos = tl.where(mask_m[:, None] & mask_d[None, :], tl.load(cos_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.), 0.)
    sin = tl.where(mask_m[:, None] & mask_d[None, :], tl.load(sin_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.), 0.)

    # process in pairs if interleaved; otherwise pair dims [d, d+offset]
    if INTERLEAVED:
        # even dims -> real, odd dims -> imaginary
        # gather real/imag
        d_even = offs_d % 2 == 0
        d_odd = ~d_even
        x_real = tl.where(d_even[None, :], x, 0.)
        x_imag = tl.where(d_odd[None, :], x, 0.)

        # shift for cos/sin
        cos_real = tl.where(d_even[None, :], cos, 0.)
        cos_imag = tl.where(d_odd[None, :], cos, 0.)
        sin_real = tl.where(d_even[None, :], sin, 0.)
        sin_imag = tl.where(d_odd[None, :], sin, 0.)

        # combine
        if CONJUGATE:
            # x_real' = x_real*cos_real + x_imag*sin_imag
            # x_imag' = x_imag*cos_imag - x_real*sin_real
            out_real = x_real * cos_real + x_imag * sin_imag
            out_imag = x_imag * cos_imag - x_real * sin_real
        else:
            # x_real' = x_real*cos_real - x_imag*sin_imag
            # x_imag' = x_imag*cos_imag + x_real*sin_real
            out_real = x_real * cos_real - x_imag * sin_imag
            out_imag = x_imag * cos_imag + x_real * sin_real

        # merge back
        out = out_real + out_imag
    else:
        # normal pairing in last dimension half/half
        half_d = D // 2
        idx_low = (offs_d < half_d)
        idx_high = ~idx_low

        x_low = tl.where(idx_low[None, :], x, 0.)
        x_high = tl.where(idx_high[None, :], x, 0.)
        cos_low = tl.where(idx_low[None, :], cos, 0.)
        cos_high = tl.where(idx_high[None, :], cos, 0.)
        sin_low = tl.where(idx_low[None, :], sin, 0.)
        sin_high = tl.where(idx_high[None, :], sin, 0.)

        if CONJUGATE:
            # x_low'  = x_low*cos_low + x_high*sin_high
            # x_high' = x_high*cos_high - x_low*sin_low
            out_low = x_low * cos_low + x_high * sin_high
            out_high = x_high * cos_high - x_low * sin_low
        else:
            # x_low'  = x_low*cos_low - x_high*sin_high
            # x_high' = x_high*cos_high + x_low*sin_low
            out_low = x_low * cos_low - x_high * sin_high
            out_high = x_high * cos_high + x_low * sin_low

        out = out_low + out_high

    # store
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


def apply_rotary(
    x, cos, sin,
    cu_seqlens=None,
    interleaved=False,
    conjugate=False
):
    import math

    b, h, m, d = x.shape
    # allocate output tensor (in-place or not, here we create a new one)
    out = tl.zeros_like(x)

    # define block sizes
    BLOCK_M = 128
    BLOCK_D = 64

    # grid
    grid_b = b
    grid_h = h
    grid_m = math.ceil(m / BLOCK_M)

    # strides
    stride_xb = x.stride(0)
    stride_xh = x.stride(1)
    stride_xm = x.stride(2)
    stride_xd = x.stride(3)

    stride_ob = out.stride(0)
    stride_oh = out.stride(1)
    stride_om = out.stride(2)
    stride_od = out.stride(3)

    stride_cb = cos.stride(0)
    stride_ch = cos.stride(1)
    stride_cm = cos.stride(2)
    stride_cd = cos.stride(3)

    stride_sb = sin.stride(0)
    stride_sh = sin.stride(1)
    stride_sm = sin.stride(2)
    stride_sd = sin.stride(3)

    rotary_kernel[(grid_b, grid_h, grid_m)](
        X_PTR=x,
        OUT_PTR=out,
        COS_PTR=cos,
        SIN_PTR=sin,
        CU_SEQLENS_PTR=cu_seqlens,
        B=b, H=h, M=m, D=d,
        STRIDE_XB=stride_xb, STRIDE_XH=stride_xh, STRIDE_XM=stride_xm, STRIDE_XD=stride_xd,
        STRIDE_OB=stride_ob, STRIDE_OH=stride_oh, STRIDE_OM=stride_om, STRIDE_OD=stride_od,
        STRIDE_CB=stride_cb, STRIDE_CH=stride_ch, STRIDE_CM=stride_cm, STRIDE_CD=stride_cd,
        STRIDE_SB=stride_sb, STRIDE_SH=stride_sh, STRIDE_SM=stride_sm, STRIDE_SD=stride_sd,
        CONJUGATE=conjugate,
        INTERLEAVED=interleaved,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
    )

    return out
