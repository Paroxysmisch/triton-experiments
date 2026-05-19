import torch
import triton
import triton.language as tl
from .linalg_utils import broadcast_batch_dims

@triton.jit
def _matrix_power_eig_kernel(a_real_ptr, a_imag_ptr, w_real_ptr, w_imag_ptr, v_real_ptr, v_imag_ptr, res_real_ptr,
                             res_imag_ptr, n, k, m, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    a_row_off = tl.program_id(axis=0) * BLOCK_SIZE_N
    v_row_off = tl.program_id(axis=0) * BLOCK_SIZE_N
    m_off = pid * BLOCK_SIZE_M
    block_last_idx = a_row_off + BLOCK_SIZE_N
    a_real_ptrs = a_real_ptr + a_row_off * n + m_off
    a_imag_ptrs = a_imag_ptr + a_row_off * n + m_off
    v_real_ptrs = v_real_ptr + v_row_off * n + m_off
    v_imag_ptrs = v_imag_ptr + v_row_off * n + m_off
    res_real_ptrs = res_real_ptr + a_row_off * n + m_off
    res_imag_ptrs = res_imag_ptr + a_row_off * n + m_off
    a_block_real = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    a_block_imag = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    v_block_real = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    v_block_imag = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    res_block_real = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    res_block_imag = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    w_real_ptrs = w_real_ptr + m_off
    w_imag_ptrs = w_imag_ptr + m_off

    for i in range(0, k):
        if i == 0:
            a_block_real = tl.load(a_real_ptrs, mask=(a_row_off < block_last_idx) & (m_off < m), other=0.).to(tl.float32)
            a_block_imag = tl.load(a_imag_ptrs, mask=(a_row_off < block_last_idx) & (m_off < m), other=0.).to(tl.float32)
            v_block_real = tl.load(v_real_ptrs, mask=(v_row_off < block_last_idx) & (m_off < m), other=0.).to(tl.float32)
            v_block_imag = tl.load(v_imag_ptrs, mask=(v_row_off < block_last_idx) & (m_off < m), other=0.).to(tl.float32)
        w_real = tl.load(w_real_ptrs).to(tl.float32)
        w_imag = tl.load(w_imag_ptrs).to(tl.float32)
        w_real_pow_k = w_real**k - k * w_real**(k - 1) * w_imag
        w_imag_pow_k = w_imag**k + k * w_real**(k - 1) * w_imag
        tmp_real = a_block_real * w_real_pow_k - a_block_imag * w_imag_pow_k
        tmp_imag = a_block_real * w_imag_pow_k + a_block_imag * w_real_pow_k
        res_block_real = tmp_real.to(res_real_ptrs.dtype.element_ty)
        res_block_imag = tmp_imag.to(res_imag_ptrs.dtype.element_ty)
        res_block_real += b_cgemm_triton_v2(a_block_real, a_block_imag, v_block_real, v_block_imag, False, True,
                                           BLOCK_SIZE_N, BLOCK_SIZE_M, n, num_stage=4, num_warps=4)
        tl.store(res_real_ptrs, res_block_real,
                 mask=(a_row_off < block_last_idx) & (m_off < m))
        tl.store(res_imag_ptrs, res_block_imag,
                 mask=(a_row_off < block_last_idx) & (m_off < m))

        tmp_real = v_block_real * w_real_pow_k - v_block_imag * w_imag_pow_k
        tmp_imag = v_block_real * w_imag_pow_k + v_block_imag * w_real_pow_k
        v_block_real = tmp_real.to(v_real_ptrs.dtype.element_ty)
        v_block_imag = tmp_imag.to(v_imag_ptrs.dtype.element_ty)
        v_block_real += b_cgemm_triton_v2(v_block_real, v_block_imag, a_block_real, a_block_imag, False, True,
                                         BLOCK_SIZE_N, BLOCK_SIZE_M, n, num_stage=4, num_warps=4)
        tl.store(v_real_ptrs, v_block_real,
                 mask=(v_row_off < block_last_idx) & (m_off < m))
        tl.store(v_imag_ptrs, v_block_imag,
                 mask=(v_row_off < block_last_idx) & (m_off < m))
        a_real_ptrs += BLOCK_SIZE_M
        a_imag_ptrs += BLOCK_SIZE_M
        v_real_ptrs += BLOCK_SIZE_M
        v_imag_ptrs += BLOCK_SIZE_M
        res_real_ptrs += BLOCK_SIZE_M
        res_imag_ptrs += BLOCK_SIZE_M
        w_real_ptrs += 1
        w_imag_ptrs += 1
        m_off += BLOCK_SIZE_M


def matrix_power_eig(A, k, *, out=None) -> torch.Tensor:
    A = A.contiguous()
    if A.ndim < 2:
        raise ValueError("Expected a matrix")
    if A.shape[-2:] != A.shape[-2:]:
        raise ValueError("The shape of input must be (*, n, n)")
    if not (isinstance(k, int) or isinstance(k, float)):
        raise ValueError("The type of exponent must be int or float")

    batch_shape, n, _ = broadcast_batch_dims(A)
    result = torch.empty(batch_shape + (n, n), dtype=A.dtype, device=A.device) \
        if out is None else out.reshape(batch_shape + (n, n))
    m = n

    if A.is_floating_point():
        MAX_BLOCK_SIZE = 128
        MIN_BLOCK_SIZE = 16
        block_size = min(max(triton.next_power_of_2(m), MIN_BLOCK_SIZE), MAX_BLOCK_SIZE)
        BLOCK_SIZE_N = triton.cdiv(n, block_size)
        BLOCK_SIZE_M = min(triton.next_power_of_2(m), block_size)
        num_stages = 3
        num_warps = 4
        grid = (BLOCK_SIZE_N,)
        L, V = torch.linalg.eig(A)
        w = torch.diag_embed(L)
        _matrix_power_eig_kernel[grid \
            ](A.real, A.imag, w.real, w.imag, V.real, V.imag, result.real, result.imag, n, k, m, BLOCK_SIZE_N,
             BLOCK_SIZE_M, num_stages=num_stages, num_warps=num_warps, extern_libs={'libdevice': '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'})
    else:
        raise RuntimeError("Only floating point matrix is supported")

    return result
