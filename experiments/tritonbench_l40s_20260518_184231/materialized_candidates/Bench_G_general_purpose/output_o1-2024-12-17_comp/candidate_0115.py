import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,        # *fp32
    rms_w_ptr,    # *fp32
    out_ptr,      # *fp32
    B, M, K,      # int32
    stride_xB, stride_xM, stride_xK,  # int32
    stride_w,                 # int32
    stride_outB, stride_outM, stride_outK,  # int32
    eps,                      # fp32
    BLOCK_N_SIZE: tl.constexpr
):
    # program ids for batch and M dimensions
    b_id = tl.program_id(0)
    m_id = tl.program_id(1)
    
    # base offset for the x/out pointers along batch and M
    x_offset_base = b_id * stride_xB + m_id * stride_xM
    out_offset_base = b_id * stride_outB + m_id * stride_outM

    # pass 1: compute sum of squares across K dimension
    sum_squares = tl.zeros([1], dtype=tl.float32)
    # loop over the K dimension in blocks of BLOCK_N_SIZE
    for k_block_start in range(0, K, BLOCK_N_SIZE):
        block_offset = k_block_start + tl.arange(0, BLOCK_N_SIZE)
        mask = block_offset < K
        # load x
        x_val = tl.load(
            x_ptr + x_offset_base + block_offset * stride_xK,
            mask=mask,
            other=0.0
        )
        # accumulate sum of squares
        sum_squares += tl.sum(x_val * x_val, axis=0)
    # compute RMS
    rms = tl.sqrt(sum_squares / K + eps)

    # pass 2: normalize and write output
    for k_block_start in range(0, K, BLOCK_N_SIZE):
        block_offset = k_block_start + tl.arange(0, BLOCK_N_SIZE)
        mask = block_offset < K
        # load x
        x_val = tl.load(
            x_ptr + x_offset_base + block_offset * stride_xK,
            mask=mask,
            other=0.0
        )
        # load RMS weights
        rms_w_val = tl.load(
            rms_w_ptr + block_offset * stride_w,
            mask=mask,
            other=0.0
        )
        # normalize and scale
        out_val = x_val / rms * rms_w_val
        # store
        tl.store(
            out_ptr + out_offset_base + block_offset * stride_outK,
            out_val,
            mask=
