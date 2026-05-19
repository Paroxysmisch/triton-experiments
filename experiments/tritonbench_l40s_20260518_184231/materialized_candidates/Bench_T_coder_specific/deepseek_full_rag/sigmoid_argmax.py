and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_argmax_kernel(input_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                          output_ptr=None):
    # starting index for the program
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # The stride represents how much we need to increase the pointer to advance 1 row
    input_ptrs = input_ptr + offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    input_values = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Compute the sigmoid function
    output_values = 1 / (1 + tl.exp(-input_values))
    # Compute the argmax
    argmax = tl.max(output_values, axis=0)
    indices = tl.argmax(output_values, axis=0)
    # Write back output to DRAM
    if output_ptr is not None:
        output_ptrs = output_ptr + offsets
        tl.store(output_ptrs, argmax, mask=mask)
    return indices

def sigmoid_argmax(input, dim=None, keepdim=False):
    if not dim and input.numel() > 1:
        raise ValueError("Expected 1D input to argmax, but got {}D input instead".format(input.dim()))
    if dim is None:
        # Flatten the input
        input = input.view(-1)
        dim = 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
    n_elements = input.numel()
    # The block size of each loop iteration is the smallest power of two greater than the number of elements
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    num_warps = 4
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = sigmoid_argmax_kernel.warmup(input, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
                                               grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // kernel.metadata.shared)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)
    num_programs = min(num_programs, 1)
    # Allocate output
    output = torch.empty((num_programs, ), dtype=torch.int64, device=input.device)
    # Create a number of persistent programs.
    indices = kernel[(num_programs, 1, 1)](
        input,
        n_elements,
        BLOCK_SIZE,
        output_ptr=output
    )
    if not keepdim:
        indices = torch.unsqueeze(indices, 0)
    if dim != 0:
        indices = torch.transpose(indices, 0, dim).contiguous()
    return indices
