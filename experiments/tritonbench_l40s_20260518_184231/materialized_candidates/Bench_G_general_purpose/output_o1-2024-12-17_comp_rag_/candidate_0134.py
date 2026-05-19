import triton
import triton.language as tl
import torch

# ---------------------------------------------------------------------------
# Document 1
# Triton kernel that takes four parameters: X (input), stride_xm, Z (output),
# stride_zn, and uses BLOCK_M, BLOCK_N values as constexpr for block size.
# ---------------------------------------------------------------------------
@triton.jit
def kernel(X, stride_xm,
           Z, stride_zn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))

ret = triton.compile(
    kernel,
    signature="*fp32,i32,*fp32,i32",
    constants={"BLOCK_M": 64, "BLOCK_N": 64}
)
print(ret.asm["ttgir"])

# ---------------------------------------------------------------------------
# Element-wise addition kernel and wrapper
# ---------------------------------------------------------------------------
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_vals = tl.load(in_ptr0 + offsets, mask=mask)
    y_vals = tl.load(in_ptr1 + offsets, mask=mask)
    out_vals = x_vals + y_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)

def add_wrapper(x, y, BLOCK_SIZE=256):
    out = torch.zeros_like(x)
    n_elements = x.numel()
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    return out
