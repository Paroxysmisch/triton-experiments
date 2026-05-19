import torch
import triton
import triton.language as tl


@triton.jit
def _symeig_impl(
    inp,
    eigenvectors,
    stride,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    inp += pid * stride
    _lambda = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _v = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    _I = tl.arange(0, BLOCK_SIZE)
    for i in range(n):
        # Compute Givens rotation
        x = tl.load(inp + i)
        y = tl.load(inp + n + i)
        r = tl.sqrt(x * x + y * y)
        c = x / r
        s = -y / r

        # Update lambda
        _lambda_old = _lambda
        _lambda = tl.where(_I == i, r, _lambda_old)
        # Update left eigenvectors
        _v_old = _v
        _v = tl.where(
            _I[:, None] == i,
            tl.stack([c, s]),
            _v_old,
        )

        # Update upper Hessenberg matrix
        # H_{i+1, i} = H_{i, i+1}^T * G
        h = tl.load(inp + i * (n + 1) + i + 1)
        g = tl.dot(tl.stack([x, y]), tl.stack([c, s]))
        tl.store(inp + i * (n + 1) + i + 1, g)
        tl.store(inp + (i + 1) * (n + 1) + i, h * s)
        tl.store(inp + i * (n + 1) + i, h * c)

    # Store results
    if eigenvectors:
        vec_stride = n * n
        vec_offset = pid * vec_stride
        out_vec = tl.device_pointer_cast(out_vec, tl.float32) + vec_offset
        tl.store(out_vec, _v.to(tl.float32))
    val_stride = n
    val_offset = pid * val_stride
    out_val = tl.device_pointer_cast(out_val, tl.float32) + val_offset
    tl.store(out_val, _lambda.to(tl.float32))


def symeig(a, eigenvectors=False, out=None):
    """
    Wrapper for the self-implemented SYMEIG kernel
    """
    a = a.contiguous()
    n = a.shape[-1]
    assert (
        a.ndim >= 2 and a.shape[-2] == a.shape[-1]
    ), "Input must be at least 2D and have equal leading and trailing dimensions"
    a_dim = a.dim()

    # Reshape input for triton kernel
    a = a.view(-1, n, n)
    batch_shape = a.shape[:-2]
    a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))
    a = a.squeeze()

    if out is not None:
        out = out.contiguous()
        out_val = out.view(-1, n)
        out_val = out_val.squeeze()
        out_vec = None
        if eigenvectors:
            out_vec = out.view(-1, n, n)
            out_vec = out_vec.squeeze()
    else:
        out_val = torch.empty((a.shape[0], a.shape[1]), device=a.device, dtype=torch.float32)
        out_vec = None
        if eigenvectors:
            out_vec = torch.empty((a.shape[0], a.shape[1], a.shape[1]), device=a.device, dtype=torch.float32)

    # Launch kernel
    def grid(meta):
        return (triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]),)

    _symeig_impl[grid](a, eigenvectors, a.stride(0), a.shape[1], a.shape[1])

    # Reshape output to match input
    out_val = out_val.reshape((*batch_shape, out_val.shape[-1]))
    if eigenvectors:
        out_vec = out_vec.reshape((*batch_shape, out_vec.shape[-2], out_vec.shape[-1]))
        return (out_val, out_vec)
    else:
        return out_val
