import torch
import triton
import triton.language as tl
from triton.runtime import driver

# Helper functions to check hardware properties
def is_hip():
    return triton.runtime.driver.active.get_current_target()[0] == 'hip'

def is_cdna():
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties.get("gcn_arch_name", "").startswith("gfx90")

# Get device properties
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

# Cache for compiled kernels
kernels = {}

@triton.jit
def mean_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=0.0)
        row_sum = tl.sum(row, axis=0)
        mean = row_sum / n_cols
        output_row_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_ptr, mean)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    # Cast input to specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    
    # Validate and process reduction dimensions
    input_dim = input.dim()
    if isinstance(dim, int):
        dim = (dim,)
    elif isinstance(dim, tuple):
        dim = tuple(sorted(dim))
    else:
        raise TypeError("dim must be int or tuple of ints")
    for d in dim:
        if d < 0 or d >= input_dim:
            raise ValueError(f"dim {d} is out of range for input of dimension {input_dim}")
    
    # Compute non-reduction and reduction dimensions
    non_reduction_dims = [d for d in range(input_dim) if d not in dim]
    reduction_dims = dim
    
    # Calculate collapsed dimensions
    n_rows = 1
    for d in non_reduction_dims:
        n_rows *= input.shape[d]
    n_cols = 1
    for d in reduction_dims:
        n_cols *= input.shape[d]
    
    # Handle empty reduction dimension
    if n_cols == 0:
        output_shape = [input.shape[d] if d in non_reduction_dims else 1 for d in range(input_dim)] if keepdim else [input.shape[d] for d in non_reduction_dims]
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        output.fill_(float('nan') if input.numel() == 0 else 0.0)
        if out is not None:
            out.copy_(output)
        return output
    
    # Reshape input to 2D tensor (n_rows, n_cols)
    input_reshaped = input.reshape(n_rows, n_cols)
    
    # Determine output shape
    output_shape = []
    if keepdim:
        for d in range(input_dim):
            output_shape.append(input.shape[d] if d in non_reduction_dims else 1)
    else:
        output_shape = [input.shape[d] for d in non_reduction_dims]
    
    # Allocate output tensor
    if out is not None:
        if tuple(out.shape) != tuple(output_shape):
            raise ValueError("out tensor has incorrect shape")
        if out.dtype != input_reshaped.dtype:
            raise ValueError("out tensor has incorrect dtype")
        output = out.reshape(-1) if not keepdim else out.reshape(n_rows, 1)
        if not output.is_contiguous():
            raise ValueError("out tensor must be contiguous")
    else:
        output = torch.empty(n_rows, dtype=input_reshaped.dtype, device=input.device)
        if keepdim:
            output = output.view(n_rows, 1)
    
    # Kernel configuration
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    
    # Pre-compile kernel and determine grid size
    kernel_key = (BLOCK_SIZE, input_reshaped.dtype)
    if kernel_key not in kernels:
        # Warmup kernel to get register usage
        warmup_output = torch.empty_like(output)
        kernel = mean_kernel.warmup(
            warmup_output,
            input_reshaped,
            input_reshaped.stride(0),
            warmup_output.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=(1,),
        )
        kernel._init_handles()
        
        # Compute occupancy
        n_regs = kernel.n_regs
        shared_mem = kernel.metadata.shared
        
        if is_hip():
            MAX_THREADS_PER_SM = properties["max_threads_per_sm"]
            max_waves_per_sm = MAX_THREADS_PER_SM // WARP_SIZE
            if is_cdna():
                num_gprs = NUM_REGS * 2
            else:
                num_gprs = NUM_REGS
            occupancy = min(num_gprs // (n_regs * WARP_SIZE), max_waves_per_sm)
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        
        occupancy = min(occupancy, SIZE_SMEM // shared_mem)
        num_programs = NUM_SM * occupancy
        kernels[kernel_key] = (kernel, num_programs)
    else:
        kernel, num_programs = kernels[kernel_key]
    
    # Launch kernel
    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        output,
        input_reshaped,
        input_reshaped.stride(0),
        output.stride(0) if output.dim() == 1 else output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    
    # Reshape output and handle 'out' tensor
    final_output = output.view(output_shape) if output.shape != output_shape else output
    if out is not None and final_output.data_ptr() != out.data_ptr():
        out.copy_(final_output)
    
    return final_output
