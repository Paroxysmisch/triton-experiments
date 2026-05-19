import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,        # *f32
    rms_w_ptr,    # *f32
    output_ptr,   # *f32
    stride_xbatch, stride_xm, stride_xn,
    stride_w,
    stride_outputbatch, stride_outputm, stride_outputn,
    N_SIZE,       # int32
    eps: float,   # float32
    BLOCK_N_SIZE: int,  # int32
    BLOCK_SIZE: tl.constexpr
):
    # program ids
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # base pointers for batch and row
    x_offset = x_ptr + pid_batch * stride_xbatch + pid_m * stride_xm
    w_offset = rms_w_ptr
    out_offset = output_ptr + pid_batch * stride_outputbatch + pid_m * stride_outputm

    # compute sum of squares
    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # keep chunk-based partial sums in registers
    sq_accum = 0.0
    num_chunks = (N_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    for i in range(num_chunks):
        n_start = i * BLOCK_SIZE
        offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_SIZE
        x_chunk = tl.load(x_offset + offsets * stride_xn, mask=mask, other=0.0)
        sq_accum += tl.sum(x_chunk * x_chunk, mask=mask)

    var = sq_accum / float(N_SIZE)
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize and write output
    for i in range(num_chunks):
        n_start = i * BLOCK_SIZE
        offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_SIZE
        x_chunk = tl.load(x_offset + offsets * stride_xn, mask=mask, other=0.0)
        w_chunk = tl.load(w_offset + offsets * stride_w, mask=mask, other=0.0)
        out_chunk = x_chunk * rstd * w_chunk
        tl.store(out_offset + offsets * stride_outputn, out_chunk, mask=mask)

def rmsnorm_triton_wrapper(x, rms_w, eps=1e-5, block_n_size=128):
    """
    x     : [batch_size, M, N]
    rms_w : [N]
    """
    B, M, N = x.shape
    output = torch.empty_like(x)
    grid = (B, M)
    rmsnorm_triton[grid](
        x, rms_w, output,
        x.stride(0), x.stride(1), x.stride(2),
        rms_w.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        N,
        eps,
        block_n_size,
        BLOCK_SIZE=block_n_size
    )
    return output
