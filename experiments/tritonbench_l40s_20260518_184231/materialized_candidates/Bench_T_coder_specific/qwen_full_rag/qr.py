import torch
import triton
import triton.language as tl


@triton.jit
def qr_kernel(W, Q, R, g, m, n, nc, block_size, stride_w_m, stride_w_n, stride_q_m, stride_q_n, stride_r_m, stride_r_n,
              num_stages, NUM_WARPS_Q, NUM_WARPS_R):
    program_id = tl.program_id(axis=0)
    if program_id < m:
        # Step 1: Compute householder vector for column i
        i = program_id
        w_i = i % n
        w_off = i * block_size + (tl.arange(0, block_size))
        q_i_off = i * block_size + tl.arange(0, block_size)
        r_ij_off = i * block_size + tl.arange(0, block_size)

        householder_start_index = tl.where(w_i < n - 1, w_i + 1, w_i)
        w_block_ptr = W + w_off + (i // block_size) * stride_w_m + (i % block_size) * stride_w_n
        w_data = tl.load(w_block_ptr, mask=w_off < n, other=float(0)).to(tl.float32)
        g_data = tl.load(g + i)[None]

        alpha = tl.sum(w_data[householder_start_index:] * w_data[householder_start_index:], axis=0)
        sigma = tl.abs(alpha) + tl.sqrt(tl.sum(tl.math.pow(w_data[householder_start_index:], 2)))

        u_1 = -tl.where(alpha >= 0., sigma, -alpha)
        rho = 1 / (sigma * u_1)
        w_bar_data = tl.where(householder_start_index < n, (w_data[householder_start_index:] - u_1) * rho, float(0))

        q_i_data = w_bar_data
        r_ij_data = w_data[:nc] * w_bar_data.conj() * rho

        q_i_block_ptr = Q + q_i_off + (i // block_size) * stride_q_m + (i % block_size) * stride_q_n
        tl.store(q_i_block_ptr, q_i_data, mask=q_i_off < n)
        r_ij_block_ptr = R + r_ij_off + (i // block_size) * stride_r_m + (i % block_size) * stride_r_n
        tl.store(r_ij_block_ptr, r_ij_data, mask=r_ij_off < n)

        # Store g
        tl.store(g + i + m, g_data * rho * u_1, mask=(i + m) < 2 * m)

    elif program_id < 2 * m:
        i = program_id - m
        tl.store(g + i, float(0))


@triton.autotune(configs=[
    triton.Config({'block_size': 32}, num_stages=3, num_warps=4),
    triton.Config({'block_size': 64}, num_stages=3, num_warps=8),
    triton.Config({'block_size': 32}, num_stages=2, num_warps=4),
    triton.Config({'block_size': 64}, num_stages=2, num_warps=8),
    triton.Config({'block_size': 32}, num_stages=1, num_warps=4),
    triton.Config({'block_size': 64}, num_stages=1, num_warps=8),
    triton.Config({'block_size': 32}),
    triton.Config({'block_size': 64})
], key=['n'])
@triton.jit
def lq_kernel(W, Q, R, g, m, n, nc, block_size, stride_w_m, stride_w_n, stride_q_m, stride_q_n, stride_r_m,
              stride_r_n, num_stages, NUM_WARPS_Q, NUM_WARPS_R):
    program_id = tl.program_id(axis=0)
    if program_id < n:
        # Step 1: Compute householder vector for row i
        i = program_id
        w_i = i % m
        w_off = i * block_size + (tl.arange(0, block_size))
        q_i_off = i * block_size + tl.arange(0, block_size)
        r_ij_off = i * block_size + tl.arange(0, block_size)

        householder_start_index = tl.where(w_i < m - 1, w_i + 1, w_i)
        w_block_ptr = W + w_off + (i // block_size) * stride_w_m + (i % block_size) * stride_w_n
        w_data = tl.load(w_block_ptr, mask=w_off < m, other=float(0)).to(tl.float32)
        g_data = tl.load(g + i)[None]

        alpha = tl.sum(w_data[householder_start_index:] * w_data[householder_start_index:], axis=0)
        sigma = tl.abs(alpha) + tl.sqrt(tl.sum(tl.math.pow(w_data[householder_start_index:], 2)))

        u_1 = -tl.where(alpha >= 0., sigma, -alpha)
        rho = 1 / (sigma * u_1)
        w_bar_data = tl.where(householder_start_index < m, (w_data[householder_start_index:] - u_1) * rho, float(0))

        q_i_data = w_bar_data
        r_ij_data = w_data[:nc] * w_bar_data.conj() * rho

        q_i_block_ptr = Q + q_i_off + (i // block_size) * stride_q_m + (i % block_size) * stride_q_n
        tl.store(q_i_block_ptr, q_i_data, mask=q_i_off < m)
        r_ij_block_ptr = R + r_ij_off + (i // block_size) * stride_r_m + (i % block_size) * stride_r_n
        tl.store(r_ij_block_ptr, r_ij_data, mask=r_ij_off < n)

        # Store g
        tl.store(g + i + m, g_data * rho * u_1, mask=(i + m) < 2 * m)

    elif program_id < 2 * m:
        i = program_id - m
        tl.store(g + i, float(0))


def qr(A, mode='reduced', *, out=None):
    A_strides = list(A.stride())
    A_dims = list(A.shape)

    if len(A_dims) < 2:
        raise RuntimeError("QR expects an input of at least 2D")

    m = A_dims[-2]
    n = A_dims[-1]
    k = min(m, n)
    q_only = False
    r_only = False
    if mode == "reduced":
        pass
    elif mode == "complete":
        raise Exception("Complete QR is not supported yet")
    elif mode == "r":
        q_only = True
    else:
        raise RuntimeError(f"Invalid value for mode: {mode}")

    if out is None:
        q_shape = A_dims
        r_shape = q_shape
        q_shape[-1] = k
        r_dtype = torch.float32 if A.dtype == torch.float32 else torch.complex64
        r_shape[-1] = n
        Q = torch.zeros(q_shape, device=A.device, dtype=A.dtype)
        R = torch.zeros(r_shape, device=A.device, dtype=r_dtype)
    else:
        assert len(out) == 2
        Q, R = out
        assert Q.shape == A.shape
        assert R.shape[-2:] == (min(Q.shape[-2], Q.shape[-1]), Q.shape[-1])

    g = torch.empty(2 * m, device=A.device, dtype=torch.float32)

    A_p = A
    if A.stride(-1) != 1:
        A_p = A.contiguous()
    n_blocks = triton.cdiv(m, 32)
    grid = (n_blocks,)
    with torch.cuda.device(A.device.index):
        qr_kernel[grid](A_p, Q, R, g, m, n, k, 32, A_p.stride(0), A_p.stride(1), Q.stride(0), Q.stride(1), R.stride(0),
                        R.stride(1), num_stages=3, NUM_WARPS_Q=8, NUM_WARPS_R=8)

    return Q, R
