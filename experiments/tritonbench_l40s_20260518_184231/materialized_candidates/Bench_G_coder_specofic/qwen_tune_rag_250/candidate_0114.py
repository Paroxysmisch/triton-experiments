import torch
import triton
import triton.language as tl

@triton.jit
def kernel(X, stride_xm, stride_xn,
           Y, stride_ym, stride_yn,
           M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid % N
    Xs = X + pid_m * stride_xm + pid_n * BLOCK_SIZE_N
    Ys = Y + pid_m * BLOCK_SIZE_M * stride_ym + pid_n * stride_yn
    mask = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M

    Xs = Xs + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_xm + tl.arange(0, BLOCK_SIZE_N)[None, :] * 1
    Ys = Ys + tl.arange(0, BLOCK_SIZE_M)[:, None] * 1 + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_yn

    X_val = tl.load(Xs, mask=mask, other=0.0)
    tl.store(Ys, X_val, mask=(mask & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N)))


def test_grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    x = torch.randn((M, N), dtype=torch.float16, device='cuda')
    y = torch.zeros((M, N), dtype=torch.float16, device='cuda')
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel[grid](x, x.stride(0), x.stride(1),
                 y, y.stride(0), y.stride(1),
                 M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
