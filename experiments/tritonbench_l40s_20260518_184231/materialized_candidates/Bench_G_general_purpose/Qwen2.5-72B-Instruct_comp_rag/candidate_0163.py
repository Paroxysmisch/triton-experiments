import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_idx = tl.program_id(0)
    
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Subtract maximum for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    
    # Optionally add a mask if mask_ptr is not None
    if mask_ptr is not None:
        mask_row = tl.load(mask_ptr + row_idx * row_stride + col_offsets, mask=mask, other=0.0)
        row_minus_max += mask_row
    
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

import torch
import triton
import triton.language as tl
import triton.compiler as tc
from triton.runtime import driver

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    if dim != -1:
        raise ValueError("Only the last dimension is supported for softmax.")
    
    if mask is not None and mask.shape != input.shape:
        raise ValueError("Mask must have the same shape as input.")
    
    n_rows, n_cols = input.shape
    
    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `input`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    num_warps = 8
    
    # Number of software pipelining stages.
    num_stages = 4
    
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        opts = {"num_warps": num_warps, "num_stages": num_stages}
        attrs = triton.compiler.AttrsDescriptor(tuple(range(6)), ()) if n_cols % 16 == 0 else None
        src = tc.ASTSource(
            fn=softmax_kernel,
            constants={"BLOCK_SIZE": BLOCK_SIZE, "num_stages": num_stages},
            signature="*fp32,*fp32,i32,i32,*fp32",
            attrs=attrs,
        )
        kernel = triton.compile(src=src, target=target, options=opts)
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)
    
    num_programs = min(num_programs, n_rows)
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Create a number of persistent programs.
    if mask is None:
        kernel[(num_programs, 1, 1)](
            output,
            input,
            input.stride(0),
            n_cols,
            None,
        )
    else:
        kernel[(num_programs, 1, 1)](
            output,
            input,
            input.stride(0),
            n_cols,
            mask,
        )
    
    return output
