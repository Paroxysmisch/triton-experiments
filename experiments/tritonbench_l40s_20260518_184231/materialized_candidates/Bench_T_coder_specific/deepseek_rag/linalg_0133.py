, max_num_waves)
            num_programs = occupancy * num_warps

        kernels[(BLOCK_SIZE, num_warps)] = (kernel, num_programs)

    # run kernel
    kernel.run(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
               num_warps=num_warps, grid=(num_programs, ))
    return y

print(softmax(torch.tensor([[1, 2, 3]], dtype=torch.float32, device='cuda')))
|system|>

User: You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Computes the inverse of a square, invertible matrix. Supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. It also supports complex number matrices. The function raises an error if the input matrix is not invertible.
Wrapper Entry Information: linalg.inv(A) -> Tensor

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of square matrices.

Returns:
    The inverse of the input matrix.
Math: A^-1
other: When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see torch.linalg.inv_ex.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 2:
Use triton language to implement a matrix multiplication operation on a 2D tensor. The kernel function 'matmul_kernel' takes 10 parameters: output_ptr (output tensor pointer), input1_ptr (first input tensor pointer), input2_ptr (second input tensor pointer), input1_row_stride (stride of first input rows), input2_row_stride (stride of second input rows), output_row_stride (stride of output rows), n_rows (number of rows), n_cols (number of columns), BLOCK_SIZE (block size for processing), and num_stages (number of software pipelining stages). The function computes the matrix multiplication for each row of the input tensors. The 'matmul' function is a wrapper that prepares the input tensors, determines the block size, number of warps, and stages, and then calls the kernel function with the appropriate grid configuration. import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def matmul_kernel(output_ptr, input1_ptr, input2_ptr, input1_row_stride, input2_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input1_ptr + row_idx * input1_row_stride
        row_start_ptr2 = input2_ptr + row_idx * input2_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input1_ptrs = row_start_ptr + col_offsets
        input2_ptrs = row_start_ptr2 + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row1 = tl.load(input1_ptrs, mask=mask, other=-float('inf'))
        row2 = tl.load(input2_ptrs, mask=mask, other=-float('inf'))
        # Compute the matrix multiplication
        output_row = tl.dot(row1, row2)
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, output_row, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def matmul(x, y):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    z = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = matmul_kernel.warmup(z, x, y, x.stride(0), y.stride(0), z.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves
