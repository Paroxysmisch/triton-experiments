import torch
import triton
import triton.language as tl
from .gelu import gelu_approx, gelu
from .utils import element_wise_kernel_configs

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['batch_dim', 'n_rows', 'n_cols'],
)
@triton.jit
def bmm_dropout_gelu_kernel(
    input_pointer, output_pointer,
    batch_dim, n_rows, n_cols,
    strides_input_batch, strides_input_row, strides_input_col,
    strides_output_batch, strides_output_row, strides_output_col,
    drop_p, seed,
    approximate_gelu: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    ):
    """
    Batch matrix multiplies two tensors, applies dropout, and GELU.

    Args:
        input_pointer: Pointer to the first input tensor.
            The first input tensor must be of shape [batch_dim, n_rows, n_cols].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, n_rows, n_cols].
        batch_dim: Batch dimension.
        n_rows: Number of rows in each matrix.
        n_cols: Number of columns in each matrix.
        strides_input_batch: Stride necessary to jump one element along the
            batch dimension of the first input tensor.
        strides_input_row: Stride necessary to jump one element along the
            row dimension of the first input tensor.
        strides_input_col: Stride necessary to jump one element along the
            column dimension of the first input tensor.
        strides_output_batch: Stride necessary to jump one element along the
            batch dimension of the container the result is written to.
        strides_output_row: Stride necessary to jump one element along the
            row dimension of the container the result is written to.
        strides_output_col: Stride necessary to jump one element along the
            column dimension of the container the result is written to.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        approximate_gelu: If True, uses tanh approximation.
            Otherwise, uses the default GELU approximation.
        BLOCK_SIZE_N: Block size for the number of rows.
        BLOCK_SIZE_M: Block size for the number of columns.
    """
    # This program processes an entire batch.
    batch_pid = tl.program_id(axis=0)

    # The seeds are generated from a global seed and the program ID so that
    # each program generates a distinct dropout mask.
    per_program_seed = seed + batch_pid

    # The block indices are distributed in a 2d grid of blocks.
    block_row_idx = tl.program_id(axis=1)
    block_col_idx = tl.program_id(axis=2)

    # Compute the block offsets.
    batch_offset = batch_pid * batch_dim
    n_offsets = block_row_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_offsets = block_col_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Load the blocks.
    input_blocks = tl.load(
        input_pointer +
        batch_offset[strides_input_batch] +
        n_offsets[:, None] * strides_input_row +
        m_offsets[None, :] * strides_input_col,
        mask=(n_offsets[:, None] < n_rows) & (m_offsets[None, :] < n_cols),
        other=0,
    )

    # Perform the batch matrix multiply.
    output_blocks = tl.dot(input_blocks, input_blocks.trans(1, 0))

    # Apply dropout.
    output_blocks = apply_dropout(
        output_blocks, drop_p, per_program_seed,
        n_offsets[:, None] * m_offsets[None, :],
    )

    # Apply GELU.
    if approximate_gelu:
        output_blocks = gelu_approx(output_blocks)
    else:
        output_blocks = gelu(output_blocks)

    # Write the output.
    tl.store(
        output_pointer +
        batch_offset[strides_output_batch] +
        n_offsets[:, None] * strides_output_row +
        m_offsets[None, :] * strides_output_col,
        output_blocks,
        mask=(n_offsets[:, None] < n_rows) & (m_offsets[None, :] < n_cols),
    )

def fused_bmm_dropout_gelu(
    input1: torch.Tensor, input2: torch.Tensor,
    p: float = 0.5, training: bool = True, inplace: bool = False,
    approximate: str = 'none',
    *,
    out = None,
) -> torch.Tensor:
    """Applies dropout and GELU to the product of two tensors."""
    # Check constraints.
    if input1.dim() != 3:
        raise ValueError('input1 must have three dimensions')
    if input2.dim() != 3:
        raise ValueError('input2 must have three dimensions')
    if input1.shape[-2:] != input2.shape[-2:]:
        raise ValueError(
            'The last two dimensions of input1 and input2 must be equal')

    batch_dim, n_rows, _ = input1.shape
    _, _, n_cols = input2.shape

    if out is None:
        out = torch.empty(
            (batch_dim, n_rows, n_cols),
            dtype=input1.dtype,
            device=input1.device,
        )
    elif out.shape != (batch_dim, n_rows, n_cols) or \
            out.dtype != input1.dtype or \
            out.device.type != input1.device.type:
        raise ValueError(
            'out must have the same shape, dtype, and device as the output')

    if inplace:
        raise NotImplementedError('inplace is not supported')

    # Prepare inputs for triton.
    input1_pointer = input1
    input2_pointer = input2
    output_pointer = out

    input1_strides = list(input1.stride())
    add_stride = list(input1.shape)[::-1]
    add_stride.pop()
    add_stride.reverse()
    add_stride = tuple(add_stride)
    input1_pointer = input1.reshape(-1).contiguous()

    # The dropout probability must be greater than 0 and less than 1.
    if p <= 0 or p >= 1:
        raise ValueError('p must be greater than 0 and less than 1')

    # Generate a random seed.
    seed = int.from_bytes(os.urandom(4), byteorder='little')

    def grid(meta):
        return (
            batch_dim,
            triton.cdiv(n_rows, meta['BLOCK_SIZE_N']),
            triton.cdiv(n_cols, meta['BLOCK_SIZE_M']),
        )

    approximate_gelu = approximate == 'tanh'

    with torch.cuda.device(input1.device.index):
        bmm_dropout_gelu_kernel[grid](
            input1_pointer, output_pointer,
            batch_dim, n_rows, n_cols,
            input1_strides[0], input1_strides[1], input1_strides[2],
            output_pointer.stride(0), output_pointer.stride(1),
            output_pointer.stride(2),
            p, seed,
            approximate_gelu,
        )

    return out
