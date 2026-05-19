import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_power_eig_kernel(V, Vinv, Lambda, k, n, num_batch, dtype, **meta):
    if dtype == torch.float32:
        EPS = 1.0e-7
    elif dtype == torch.float64:
        EPS = 1.0e-15
    else:
        EPS = 1.0e-7

    r = tl.zeros([num_batch, n], dtype=tl.float32)
    for i in range(num_batch):
        for j in range(0, n, 32):
            idx = j + tl.arange(0, 32)
            mask = idx < n
            x = tl.load(V + i * n + idx, mask=mask).to(tl.float32)
            y = tl.load(Lambda + i * n + idx, mask=mask).to(tl.float32)
            y = tl.where(tl.abs(y) > EPS, y**k, 0.0)
            z = tl.dot(y, x, allow_tf32=False)
            r = tl.store(r + i * n + idx, z, mask=mask)

    tl.debug_barrier()
    for i in range(num_batch):
        for j in range(0, n, 32):
            idx = j + tl.arange(0, 32)
            mask = idx < n
            x = tl.load(r + i * n + idx, mask=mask).to(tl.float32)
            y = tl.load(Vinv + i * n + idx, mask=mask).to(tl.float32)
            z = tl.dot(x, y, allow_tf32=False)
            tl.store(V + i * n + idx + meta["offset"], z, mask=mask)

def matrix_power_eig(A, k, *, out=None) -> torch.Tensor:
    dtype = A.dtype
    assert dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]
    device = A.device
    num_batch = A.shape[:-2]
    n = A.shape[-1]
    A = A.reshape([-1, n, n])
    if dtype in [torch.complex64, torch.complex128]:
        v, V = torch.linalg.eig(A)
    else:
        v, V = torch.linalg.eigh(A)
    Vinv = torch.inverse(V)
    if out is None:
        out = torch.empty_like(V @ Vinv, dtype=dtype, device=device)
    else:
        assert out.shape == V.shape
        assert out.dtype == dtype
        assert out.device == device
        out = out.reshape([-1, n, n])
    Lambda = torch.zeros_like(V, device=device, dtype=dtype)
    for i in range(num_batch):
        Lambda[i * n:(i + 1) * n, i * n:(i + 1) * n] = torch.diag(v[i * n:(i + 1) * n] ** k)
    _matrix_power_eig_kernel[(1,)](
        out, Vinv, Lambda, k, n, num_batch, dtype, offset=n * num_batch
    )
    return out.reshape([*list(num_batch), n, n])
