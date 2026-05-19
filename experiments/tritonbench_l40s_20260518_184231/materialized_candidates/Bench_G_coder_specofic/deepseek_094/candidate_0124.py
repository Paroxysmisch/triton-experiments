import triton
import triton.language as tl

@triton.autotune(params=[triton.Param("N", values=[1, 2, 4, 8, 16]), triton.Param("ND", values=[1, 2, 4, 8, 16])])
@triton.jit
def logsumexp_fwd_kernel(x_p, z_p, B, D, N, ND, HAS_SCALE):
    pid_i, pid_j = tl.program_id(0), tl.program_id(1)
    i_n, i_d = pid_i, pid_j // N
    o_d, m_d = pid_j % N, pid_i // ND
    b_x = tl.load(x_p + [i_n * D + o_d])
    b_m = tl.max(b_x)
    if HAS_SCALE:
        b_x = b_x - b_m
    z = tl.sum(tl.exp(b_x))
    z = tl.log(z)
    z = z + b_m
    tl.store(z_p + [i_n * ND + o_d], z)

def logsumexp_fwd(x, out_dtype=None):
    x = x.reshape(-1, x.shape[-1])
    N, D = x.shape
    B = 1
    ND = D // B
    z = np.empty((N, ND), dtype=np.float32)
    grid = (N, ND)
    logsumexp_fwd_kernel[grid](x, z, B, D, N, ND, HAS_SCALE=D % B != 0)
    z = np.sum(z, axis=1)
    if out_dtype is not None:
        z = z.astype(out_dtype)
    return z
