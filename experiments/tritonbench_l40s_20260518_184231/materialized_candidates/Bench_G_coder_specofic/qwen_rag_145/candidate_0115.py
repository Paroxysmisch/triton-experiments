import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   out_ptr: tl.pointer_type,
                   x_stride0: tl.uint32,
                   x_stride1: tl.uint32,
                   N_SIZE: tl.uint32,
                   eps: tl.float32,
                   BLOCK_N_SIZE: tl.constexpr):
    # Instances are assigned unique program ids using `program_id` method.
    pgid = tl.program_id(0)
    batch_id = pgid // x_stride0
    m_id = pgid % x_stride0
    offsets = tl.arange(0, BLOCK_N_SIZE)
    mask = offsets < N_SIZE

    # Load rms_weights
    rms_weight = tl.load(rms_w_ptr + batch_id)

    # Compute pointers
    x_ptrs = x_ptr + batch_id * x_stride0 * x_stride1 + m_id * x_stride1 + offsets
    out_ptrs = out_ptr + batch_id * x_stride0 * x_stride1 + m_id * x_stride1 + offsets

    # Load x & squared
    x_vals = tl.load(x_ptrs, mask=mask, other=0)
    x_squared = x_vals * x_vals

    # Compute RMS using the `sum` function
    rms_x = tl.sum(x_squared) / N_SIZE
    rms_x = tl.sqrt(rms_x + eps)

    # Normalize x & store
    norm_vals = x_vals / rms_x
    output_vals = norm_vals * rms_weight
    tl.store(out_ptrs, output_vals, mask=mask)

def rmsnorm_wrapper(x, rms_weights, out, N_SIZE, eps=1e-5, BLOCK_N_SIZE=512, num_warps=8):
    # Get shapes & strides
    batch, M, K = x.shape
    x_stride0 = x.stride(0) * M
    x_stride1 = x.stride(1)
    # Number of programs should be batch * M
    n_pg = batch * M

    # Organise inputs
    x_ptr = x.data_ptr()
    rms_w_ptr = rms_weights.data_ptr()
    out_ptr = out.data_ptr()

    # Invoke kernel
    rmsnorm_triton[(n_pg,)](x_ptr, rms_w_ptr, out_ptr, x_stride0, x_stride1, N_SIZE, eps, n_warps=num_warps, BLOCK_N_SIZE=BLOCK_N_SIZE)
