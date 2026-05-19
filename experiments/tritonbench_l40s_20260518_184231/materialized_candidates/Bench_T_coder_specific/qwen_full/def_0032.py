import torch
import triton
import triton.language as tl

@triton.jit
def _eig(
    A, stride_za, stride_ha,  #
    w, stride_zw, stride_hw,  #
    v, stride_zv, stride_hv,  #
    n,  #
    # Return code
    ret,  #
    # Meta-params
    BLOCK_SIZE: tl.constexpr,  #
):
    # Tile indices
    z = tl.program_id(0)
    # The remainder is handled by the loop
    # i = tl.program_id(1)
    # j = tl.program_id(2)

    # Offset pointers to the first element of the z-th batch
    A = A + z * stride_za
    w = w + z * stride_zw
    v = v + z * stride_zv

    # Block indices
    i = tl.arange(0, BLOCK_SIZE)
    # Row index
    # i = i + (tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    # Col index
    # j = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Pointer arithmetic on the block
    # Advance the pointer to the correct row
    A = A + i * stride_ha
    w = w + i * stride_hw
    v = v + i * stride_hv
    # The remainder is handled in the loop
    # Advance the pointer to the correct column
    # A = A + j
    # w = w + j
    # v = v + j

    # Compute the eigendecomposition
    # This is equivalent to the eigen decomposition of a batch of matrices
    # The eigenvectors are normalized to have norm 1
    _eig_(A, w, v, n, ret)

def _eig(A, *, out=None):
    # Error if A is not a matrix
    if A.dim() < 2:
        raise ValueError("expected a matrix")
    if A.dim() == 2:
        batch = ()
        n = A.size(0)
    else:
        batch = A.shape[:-2]
        n = A.size(-2)
    if A.size(-1) != n:
        raise ValueError("expected a square matrix")

    if out is None:
        w = torch.empty(batch + (n,), dtype=A.dtype, device=A.device)
        v = torch.empty(batch + (n, n), dtype=A.dtype, device=A.device)
    else:
        w, v = out
        if w.shape != batch + (n,):
            raise ValueError("shape of w does not match the expected shape")
        if v.shape != batch + (n, n):
            raise ValueError("shape of v does not match the expected shape")

    # The value of the return code is checked in Python
    ret = torch.zeros(batch, dtype=torch.int32, device=A.device)

    # Prepare inputs for Triton kernel
    # Pointers
    A = A.data_ptr()
    w = w.data_ptr()
    v = v.data_ptr()
    # Strides
    stride_za = A.stride(-3) if A.dim() > 2 else 0
    stride_ha = A.stride(-2)
    stride_zw = w.stride(-2) if w.dim() > 1 else 0
    stride_hw = w.stride(-1)
    stride_zv = v.stride(-3) if v.dim() > 2 else 0
    stride_hv = v.stride(-2)

    # Run the kernel
    _eig[(batch,)](
        A, stride_za, stride_ha,  #
        w, stride_zw, stride_hw,  #
        v, stride_zv, stride_hv,  #
        n,  #
        ret,  #
        # Meta-params
        BLOCK_SIZE=triton.next_power_of_2(n),  #
    )
    # Check return code
    if ret.item() != 0:
        raise RuntimeError("eig failed")

    return w, v
