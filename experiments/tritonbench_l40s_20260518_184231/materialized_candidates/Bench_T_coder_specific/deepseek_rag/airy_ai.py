import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def airy_ai_kernel(
        A_ptr,
        O_ptr,
        M,
        BLOCK_SIZE: tl.constexpr
):
    # Get the row index for the current program
    row_id = tl.program_id(axis=0)
    # Calculate offsets for the current row
    offsets = row_id*M + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = tl.arange(0, BLOCK_SIZE) < M

    # Load the row elements from A_ptr with masking
    a = tl.load(A_ptr + offsets, mask=mask, other=-float('inf'))
    # Compute the Airy function Ai
    op = tl.airy.ai(a)

    # Store the result in O_ptr with masking
    tl.store(O_ptr + offsets, op, mask=mask)


def airy_ai(A: torch.Tensor, out: torch.Tensor = None):
    # Get the dimensions of the input tensor
    rows, cols = A.shape
    # Prepare an output tensor on the same device
    if out is None:
        output = torch.empty(size=A.shape).to(device)
    else:
        output = out

    # Ensure both input and output tensors are on the GPU
    assert A.is_cuda and output.is_cuda, 'One of the matrix is not on GPU'

    # Block size will be equal to number of columns
    # so that every row is operated in one block
    BLOCK_SIZE = triton.next_power_of_2(cols)
    print(f'Block size: {BLOCK_SIZE}, grid: {(rows,)}')

    # Launch the Triton kernel
    airy_ai_kernel[(rows,)](
        A_ptr=A, O_ptr=output, M=cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
