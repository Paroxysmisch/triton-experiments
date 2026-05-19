import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def zeta_kernel(x_ptr, q_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the index for the current program
    idx = tl.program_id(axis=0)
    
    # Calculate offsets for the current element
    offsets = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < N

    # Load the x and q values
    x = tl.load(x_ptr + idx, mask=mask)
    q = tl.load(q_ptr + idx, mask=mask)

    # Initialize the output
    result = tl.zeros_like(x)

    # Compute the Hurwitz zeta function
    for k in range(100):  # Limit the series to 100 terms for convergence
        term = 1 / ((k + q) ** x)
        result += term

    # Store the result in out_ptr
    tl.store(out_ptr + idx, result, mask=mask)

def zeta(input: torch.Tensor, other: torch.Tensor, *, out=None) -> torch.Tensor:
    # Get the number of elements
    N = input.numel()
    
    # Prepare an output tensor on the same device
    if out is None:
        out = torch.empty_like(input)

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and other.is_cuda and out.is_cuda, 'One of the tensors is not on GPU'

    # Block size will be equal to the number of elements
    BLOCK_SIZE = triton.next_power_of_2(N)
    print(f'Block size: {BLOCK_SIZE}, grid: {(N,)}')

    # Launch the Triton kernel
    zeta_kernel[(N,)](
        x_ptr=input, q_ptr=other, out_ptr=out, N=N, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
