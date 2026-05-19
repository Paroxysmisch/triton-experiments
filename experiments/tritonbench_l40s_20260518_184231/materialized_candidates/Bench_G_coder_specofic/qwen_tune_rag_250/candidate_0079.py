import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(A, stride_am, stride_an, B, stride_bm, stride_bn, M, N, absmax_inv, 
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_N)
    pid_n = pid % (N // BLOCK_N)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_a = offs_am[:, None] * stride_am + offs_bn[None, :] * stride_an
    a = tl.load(A + offs_a)
    absmax = tl.abs(a)
    absmax = tl.maximum(tl.max(absmax, axis=1), 1e-6)
    absmax_inv = 127. / absmax
    a_q = (a * absmax_inv + 0.5).to(tl.int8)
    offs_bm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = offs_bm[:, None] * stride_bm + offs_bn[None, :] * stride_bn
    tl.store(B + offs_b, a_q)

def quantize_global_transpose(A, B, absmax, absmax_inv):
    M, N = A.shape
    grid_fn = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    _quantize_global_transpose[grid_fn](A, B, absmax, absmax_inv, BLOCK_M=128, BLOCK_N=128, GROUP_M=8)
