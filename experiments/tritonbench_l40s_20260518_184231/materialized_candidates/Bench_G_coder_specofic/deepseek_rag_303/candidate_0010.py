import triton
import triton.language as tl

@triton.jit
def mv_kernel(
    A,
    B,
    C,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Calculate the row-offset of A and B
    offset_am = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offset_ak = tl.arange(0, BLOCK_N)
    offset_b = tl.arange(0, BLOCK_N)
    # Load A block
    a = tl.load(A + offset_am[:, None] * M + offset_ak[None, :])
    # Broadcast B and do element-wise multiplication
    # this will load new data at each iteration
    b = tl.load(B + offset_b)
    # Compute partial dot-product
    acc = tl.dot(a, b)
    # Sum partial acc results for rows
    acc = tl.sum(acc, axis=1)
    # Write back result
    offset_cm = tl.arange(0, BLOCK_M) + pid * BLOCK_M
    tl.store(C + offset_cm, acc)

def mv(A, B):
    assert A.shape[1] == B.shape[0], "incompatible dimensions"
    M, N = A.shape
    C = torch.empty((M,), device=A.device, dtype=A.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch.cuda.device(A.device):
        mv_kernel[grid](A, B, C, M, N)
    return C
