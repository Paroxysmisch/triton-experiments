import torch
import triton
import triton.language as tl
from triton import runtime
from triton.runtime import driver

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Add a check for input's layout being sparse
    if isinstance(input, torch.Tensor) and input.layout == torch.sparse_coo:
        return _addmm_sparse_cs_cs(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
    elif isinstance(mat1, torch.Tensor) and mat1.layout == torch.sparse_coo:
        return _addmm_sparse_cs_cs(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
    elif isinstance(mat2, torch.Tensor) and mat2.layout == torch.sparse_coo:
        return _addmm_sparse_cs_cs(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
    else:
        return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

@triton.jit
def addmm_kernel(input_ptr, mat1_ptr, mat2_ptr, beta, alpha, output_ptr, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """
    Kernel for computing the matmul of a sparse matrix and a dense matrix.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(K, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % N
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % K
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = input_ptr + offs_am[:, None] * K + offs_k[None, :]
    b_ptrs = mat1_ptr + (offs_k[:, None] * N + offs_am[None, :])
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = output_ptr + offs_cm[:, None] * K + offs_cn[None, :]
    mask = (offs_cm[:, None] < N) & (offs_cn[None, :] < K)
    x = tl.load(c_ptrs, mask=mask, other=0.0)
    x *= beta
    x += tl.dot(c, alpha * mat2_ptr + offs_bn[None, :] * K + offs_cm[:, None])
    tl.store(c_ptrs, x.to(tl.float32), mask=mask)

def _addmm_sparse_cs_cs(input, mat1, mat2, beta=1, alpha=1, out=None):
    device = input.device
    if out is None:
        out = torch.empty((input.shape[0], mat2.shape[1]), device=device, dtype=torch.float16)
    else:
        assert out.dtype == torch.float16, "Output tensor must have dtype torch.float16"
        assert out.shape == (input.shape[0], mat2.shape[1]), "Output tensor must be of shape (input.shape[0], mat2.shape[1])"
        assert out.device == device, "Output tensor must be on the same device as input and mat2 tensors"
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions between mat1 and mat2"
    assert input.is_sparse_csr, "Input tensor must be a sparse CSR tensor"
    assert mat1.is_sparse_coo, "mat1 tensor must be a sparse COO tensor"
    assert mat2.is_contiguous(), "mat2 tensor must be contiguous"
    mat1 = mat1.coalesce()
    m, k = input.shape
    _, n = mat2.shape
    grid = lambda META: (triton.cdiv(META["N"], META["BLOCK_N"]) * triton.cdiv(META["K"], META["BLOCK_K"]) * META["GROUP_M"],)
    addmm_kernel[grid](
        input,
        mat1.values_ptr(),
        mat2,
        beta,
        alpha,
        out,
        m,
        k,
        GROUP_M=4,
    )
    return out
