import triton
import triton.language as tl
import torch

@triton.jit
def signbit_kernel(
    output_ptr,
    input_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program will process a block of elements
    pid = tl.program_id(axis=0)
    coords = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = coords + offsets
    valid_mask = indices < N

    # Load input values
    input_values = tl.load(input_ptr + indices, mask=valid_mask)

    # Check if the sign bit is set
    sign_bits = tl.bitwise_and(input_values, 0x8000000000000000)

    # Store results in output buffer
    tl.store(output_ptr + indices, sign_bits != 0, mask=valid_mask)

def signbit(input, *, out=None):
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    N = input.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate memory for the output tensor
    output = out.new_empty((N,), dtype=torch.bool)

    # Launch the Triton kernel
    signbit_kernel[(num_blocks,)](
        output.data_ptr(),
        input.data_ptr(),
        N,
        BLOCK_SIZE,
        num_warps=4,
    )

    return output
