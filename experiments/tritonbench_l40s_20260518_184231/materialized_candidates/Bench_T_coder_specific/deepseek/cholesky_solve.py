@triton.jit
def _cholesky_solve(Y, L, out,
                    BLOCK_SIZE: tl.constexpr,
                    N: tl.constexpr):
    pid = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    m = tl.program_id(axis=2)

    # Load inputs
    Y_ = Y[pid, :, :]
    L_ = L[pid, :, :]

    # Compute solution
    for i in range(N):
        for j in range(i, N):
            tmp = L_[i, j]
            for k in range(i):
                tmp -= L_[k, j] * L_[i, k]
            L_[i, j] = tmp / L_[i, i]
        for j in range(i, N):
            tmp = Y_[i, j]
            for k in range(i):
                tmp -= L_[k, j] * L_[i, k]
            Y_[i, j] = tmp / L_[i, i]

    # Store output
    out[pid, :, :] = Y_
