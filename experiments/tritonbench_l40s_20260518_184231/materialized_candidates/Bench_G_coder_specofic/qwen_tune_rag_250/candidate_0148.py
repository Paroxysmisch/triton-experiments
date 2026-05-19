import triton
import triton.language as tl

# triton kernel
@triton.jit
def mul2_kernel(X, stride_xm, Z, stride_zn, n_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    Xs = X + offsets * stride_xm
    Zs = Z + offsets * stride_zn
    x = tl.load(Xs, mask=mask)
    z = x * 2
    tl.store(Zs, z, mask=mask)

def triton_mul2(X):
    shape = X.shape
    assert len(shape) == 1, "only accept 1D tensor now"
    n_elements = shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    Z = torch.empty_like(X)
    mul2_kernel[grid](X, Z.stride(0), Z, X.stride(0), n_elements, BLOCK_SIZE=1024)
    return Z

# triton kernel
@triton.jit
def mul2_inplace_kernel(X, stride_xm, n_elements,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    Xs = X + offsets * stride_xm
    x = tl.load(Xs, mask=mask)
    z = x * 2
    tl.store(Xs, z, mask=mask)

def triton_mul2_inplace(X):
    shape = X.shape
    assert len(shape) == 1, "only accept 1D tensor now"
    n_elements = shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](X, X.stride(0), n_elements, BLOCK_SIZE=1024)
