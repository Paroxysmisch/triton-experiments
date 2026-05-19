import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel for matrix-vector multiplication
@triton.jit
def mv_kernel(A, B, C, M, N, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the block of C that this program will handle
    block_start = pid * BLOCK_N
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = offsets < N
    # Load the block of C
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        block_m = m + tl.arange(0, BLOCK_M)
        # Load a block of A and B matrices, do matrix multiplication
        a = tl.load(A + offsets[:, None] * N + block_m[None, :], mask=mask[:, None] & (block_m[None, :] < M), other=0.0)
        b = tl.load(B + block_m, mask=block_m < M, other=0.0)
        acc += tl.sum(a * b, 1)
    # Write back the result
    tl.store(C + offsets, acc, mask=mask)

# Wrapper function for calling the Triton kernel
def mv(
    input: torch.Tensor,
    vec: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    *,
    skip_checks: bool = False
) -> torch.Tensor:
    assert input.dim() >= 2 and vec.dim() == 1, "Only accepts 2D or higher input and 1D vec"
    assert input.size(-1) == vec.size(0), "Last dim of input must match vec"
    if not skip_checks:
        assert (
            input.is_cuda and vec.is_cuda
        ), "Both input and vec must be CUDA tensors if custom Triton kernel is used"
    if out is None:
        out = torch.empty(input.size()[:-1], device=input.device, dtype=input.dtype)
    else:
        assert out.shape == input.shape[:-1] and out.dtype == input.dtype and out.device == input.device
    # Make sure that the input tensor is contiguous in the trailing dimensions
    input = input.contiguous()
    # Run the Triton kernel
    N, M = input.shape
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch.cuda.device(input.device):
        mv_kernel[grid](input, vec, out, M, N)
    if bias is not None:
        assert bias.is_cuda and bias.ndim == 1, "Bias must be a 1D CUDA tensor"
        assert bias.numel() == out.numel(), "The shape of bias must match the output"
        out += bias
    return out
