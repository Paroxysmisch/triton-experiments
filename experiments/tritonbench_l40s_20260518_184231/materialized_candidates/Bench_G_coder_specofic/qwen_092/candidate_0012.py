import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(X, Y, W, B, Mean, RSTD, stride, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    for i in range(block_start, block_end):
        mean = 0.0
        var = 0.0
        for j in range(M):
            x = X[i * stride + j]
            mean += x
        mean /= M
        for j in range(M):
            x = X[i * stride + j]
            var += (x - mean) ** 2
        var /= M
        var += 1e-5
        rstd = 1.0 / tl.sqrt(var)
        for j in range(M):
            x = X[i * stride + j]
            y = (x - mean) * rstd * W[j] + B[j]
            Y[i * stride + j] = y
        Mean[i] = mean
        RSTD[i] = rstd

@triton.jit
def _layer_norm_backward_kernel(DX, DY, DW, DB, X, Y, W, B, Mean, RSTD, stride, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    for i in range(block_start, block_end):
        mean = Mean[i]
        rstd = RSTD[i]
        for j in range(M):
            dy = DY[i * stride + j]
            dx = dy * rstd * W[j]
            DX[i * stride + j] = dx
            dw = dy * (Y[i * stride + j] - mean) * rstd
            DW[j] += dw
            db = dy
            DB[j] += db
