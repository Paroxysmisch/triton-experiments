import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    # Subtract maximum for numerical stability
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(inp, dtype=None, is_half=False):
    if dtype is None:
        dtype = inp.dtype
    if not is_half:
        dtype = getattr(tl, dtype)
    n_cols = inp.shape[-1]
    # We want to use the smallest power of two greater than n_cols to ensure we
    # can fit each row in a single block
    BLOCK_SIZE = int(2 ** (n_cols.bit_length() + 1))
    # The grid is one block per row
    grid = (inp.shape[0],)
    # We need an output holder since triton kernels cannot return tensors
    out = torch.empty_like(inp, dtype=dtype)
    # Launch the triton kernel
    softmax_kernel[grid](
        out,
        inp,
        inp.stride(0),
        out.stride(0),
        n_cols,
        num_warps=min(32, (BLOCK_SIZE + 1023) // 1024),
        constants={"BLOCK_SIZE": BLOCK_SIZE},
    )
    return out
