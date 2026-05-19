import torch
import triton
import triton.language as tl
from torch import Tensor
from .utils import maybe_profile

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32),
    ],
    key=["n_elements"],
)
@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    PRE_LOAD: tl.constexpr = False,
):
    # Map the program id to the row of the input it should compute.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def _grid(meta):
    return (triton.cdiv(meta["n_rows"], meta["BLOCK_SIZE"]), 1, 1)


def softmax(input: Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None) -> Tensor:
    f_name = "softmax"
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    input_dim = dim % input.ndim
    assert input.shape[input_dim] > 0, "Softmax dim cannot be empty"

    if dtype is None:
        dtype = input.dtype
    elif dtype != input.dtype:
        input = input.to(dtype)

    # Flatten the input tensors to a 2D tensor where each row represents a slice
    # across the dimension over which we're applying softmax.
    shape = list(input.shape)
    n_rows = shape.pop(input_dim)
    shape.insert(0, n_rows)
    n_cols = prod(shape)
    input = input.reshape(n_rows, n_cols)
    output = torch.empty_like(input)

    with torch.cuda.device(input.device):
        _softmax_kernel[(_grid)](
            output,
            input,
            input.stride(0),
            output.stride(0),
            n_rows,
            n_cols,
        )

    return output.reshape_as(input)
