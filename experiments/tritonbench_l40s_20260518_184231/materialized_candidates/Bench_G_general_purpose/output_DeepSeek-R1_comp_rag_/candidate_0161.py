import torch
import triton
import triton.language as tl
import triton.compiler as tc
from triton.runtime import driver

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_rows, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        if mask_ptr != 0:
            mask_row_start_ptr = mask_ptr + row_idx * input_row_stride
            mask_ptrs = mask_row_start_ptr + col_offsets
            mask_values = tl.load(mask_ptrs, mask=mask, other=0)
            row_minus_max += mask_values
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    assert dim == -1 or dim == input.dim() - 1, "Softmax must be applied along the last dimension"
    input = input.contiguous()
    if input.dim() != 2:
        n_rows = input.numel() // input.size(-1)
        n_cols = input.size(-1)
        input = input.view(n_rows, n_cols)
    else:
        n_rows, n_cols = input.shape
    if mask is not None:
        assert mask.shape == input.shape, "Mask must have the same shape as input"
        mask = mask.contiguous()
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if BLOCK_SIZE >= 2048 else 3
    mask_ptr = mask.data_ptr() if mask is not None else 0
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        opts = {"num_warps": num_warps, "num_stages": num_stages}
        attrs = triton.compiler.AttrsDescriptor(tuple(range(7)), ()) if n_cols % 16 == 0 else None
        src = tc.ASTSource(
            fn=softmax_kernel,
            constants={"BLOCK_SIZE": BLOCK_SIZE, "num_stages": num_stages},
            signature="*fp32,*fp32,i32,i32,i32,i32,*fp32",
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
    output = torch.empty_like(input)
    kernel[(num_programs, 1, 1)](
        output,
        input,
        input.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        mask_ptr,
    )
    return output
