import torch
import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def _softmax(output_ptr, input_ptr, input_row_stride, input_col_stride, output_row_stride, output_col_stride,
             n_rows, n_cols, n_depth, BLOCK_SIZE: tl.constexpr, LOG: tl.constexpr, CAUSAL: tl.constexpr,
             MASK_TYPE: tl.constexpr, IS_FP16: tl.constexpr, num_stages: tl.constexpr):
    # Starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Starting depth of the program
        depth_start = tl.program_id(1)
        depth_step = tl.num_programs(1)
        for depth_idx in tl.range(depth_start, n_depth, depth_step, num_stages=num_stages):
            # The stride represents how much we need to increase the pointer to advance 1 row
            row_start_ptr = input_ptr + row_idx * input_row_stride + depth_idx * input_col_stride
            # The block size is the next power of two greater than n_cols, so we can fit each
            # row in a single block
            col_offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = row_start_ptr + col_offsets
            # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
            mask = col_offsets < n_cols
            row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
            # Subtract maximum for numerical stability
            row_minus_max = row - tl.max(row, axis=0)
            # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)
            softmax_output = numerator / denominator
            if LOG:
                softmax_output = tl.log(softmax_output)
            if CAUSAL:
                causal_mask = col_offsets < depth_idx
                softmax_output = tl.where(causal_mask, softmax_output, 0.0)
            if MASK_TYPE == 1:
                # Apply a custom mask (e.g., attention mask)
                mask_ptr = mask_ptr + row_idx * mask_row_stride + depth_idx * mask_col_stride
                mask = tl.load(mask_ptr, mask=mask, other=0.0)
                softmax_output = softmax_output * mask
            # Write back output to DRAM
            output_row_start_ptr = output_ptr + row_idx * output_row_stride + depth_idx * output_col_stride
            output_ptrs = output_row_start_ptr + col_offsets
            tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x, log=False, causal=False, mask=None):
    n_rows, n_cols, n_depth = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Number of warps and stages
    num_warps = 8
    num_stages = 4

    # Pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        opts = {"num_warps": num_warps, "num_stages": num_stages}
        attrs = triton.compiler.AttrsDescriptor(tuple(range(6)), ()) if n_cols % 16 == 0 else None
        src = triton.compiler.ASTSource(
            fn=_softmax,
            constants={"BLOCK_SIZE": BLOCK_SIZE, "LOG": log, "CAUSAL": causal, "MASK_TYPE": 0 if mask is None else 1, "IS_FP16": x.dtype == torch.float16, "num_stages": num_stages},
            signature="*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32",
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

    num_programs = min(num_programs, n_rows * n_depth)

    # Allocate output
    y = torch.empty_like(x)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        n_rows,
        n_cols,
        n_depth,
        BLOCK_SIZE,
    )
    return y

@triton.jit
def _softmax_backward(grad_output_ptr, output_ptr, input_ptr, input_row_stride, input_col_stride, output_row_stride, output_col_stride,
                      n_rows, n_cols, n_depth, BLOCK_SIZE: tl.constexpr, LOG: tl.constexpr, CAUSAL: tl.constexpr,
                      MASK_TYPE: tl.constexpr, IS_FP16: tl.constexpr, num_stages: tl.constexpr):
    # Starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Starting depth of the program
        depth_start = tl.program_id(1)
        depth_step = tl.num_programs(1)
        for depth_idx in tl.range(depth_start, n_depth, depth_step, num_stages=num_stages):
            # The stride represents how much we need to increase the pointer to advance 1 row
            row_start_ptr = input_ptr + row_idx * input_row_stride + depth_idx * input_col_stride
            output_start_ptr = output_ptr + row_idx * output_row_stride + depth_idx * output_col_stride
            grad_output_start_ptr = grad_output_ptr + row_idx * output_row_stride + depth_idx * output_col_stride
            # The block size is the next power of two greater than n_cols, so we can fit each
            # row in a single block
            col_offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = row_start_ptr + col_offsets
            output_ptrs = output_start_ptr + col_offsets
            grad_output_ptrs = grad_output_start_ptr + col_offsets
            # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
            mask = col_offsets < n_cols
            row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
            output = tl.load(output_ptrs, mask=mask, other=0.0)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
            # Compute gradients
            grad_input = output * (grad_output - tl.sum(output * grad_output, axis=0))
            if LOG:
                grad_input = grad_input / output
            if CAUSAL:
                causal_mask = col_offsets < depth_idx
                grad_input = tl.where(causal_mask, grad_input, 0.0)
            if MASK_TYPE == 1:
                # Apply a custom mask (e.g., attention mask)
                mask_ptr = mask_ptr + row_idx * mask_row_stride + depth_idx * mask_col_stride
                mask = tl.load(mask_ptr, mask=mask, other=0.0)
                grad_input = grad_input * mask
            # Write back output to DRAM
            grad_input_ptrs = input_ptrs
            tl.store(grad_input_ptrs, grad_input, mask=mask)
