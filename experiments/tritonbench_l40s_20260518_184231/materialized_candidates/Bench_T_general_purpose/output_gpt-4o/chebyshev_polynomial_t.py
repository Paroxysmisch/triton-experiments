import triton
import triton.language as tl
import torch

@triton.jit
def chebyshev_polynomial_kernel(input_ptr, n_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input and n values
    input = tl.load(input_ptr + offsets, mask=offsets < numel, other=0.0)
    n = tl.load(n_ptr + offsets, mask=offsets < numel, other=0)

    # Initialize output
    out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Compute Chebyshev polynomial
    for i in range(BLOCK_SIZE):
        if offsets[i] < numel:
            if n[i] == 0:
                out[i] = 1.0
            elif n[i] == 1:
                out[i] = input[i]
            elif n[i] < 6 or abs(input[i]) > 1.0:
                Tnm1 = 1.0  # T_0
                Tn = input[i]  # T_1
                for j in range(2, n[i] + 1):
                    Tnp1 = 2 * input[i] * Tn - Tnm1
                    Tnm1 = Tn
                    Tn = Tnp1
                out[i] = Tn
            else:
                out[i] = tl.cos(n[i] * tl.acos(input[i]))

    # Store result
    tl.store(out_ptr + offsets, out, mask=offsets < numel)

def chebyshev_polynomial_t(input, n, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    if not isinstance(n, torch.Tensor):
        raise ValueError("n must be a torch.Tensor")

    # Check dimensions
    if input.shape != n.shape:
        raise ValueError("Input and n must have the same shape")

    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)

    # Launch Triton kernel
    numel = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    chebyshev_polynomial_kernel[grid](input, n, out, numel, BLOCK_SIZE=BLOCK_SIZE)

    return out
