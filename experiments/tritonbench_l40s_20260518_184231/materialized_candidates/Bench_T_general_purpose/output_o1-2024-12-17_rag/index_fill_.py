import torch
import triton
import triton.language as tl

@triton.jit
def _index_fill_kernel_2d(
    data_ptr,          # Pointer to the data (self tensor)
    index_ptr,         # Pointer to the index tensor
    value,             # The fill value
    stride0,           # Stride for the 0th dimension
    stride1,           # Stride for the 1st dimension
    n_index,           # Number of indices
    n_other,           # Size of the other dimension
    dim,               # The dimension along which to fill (0 or 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a mask to ensure we stay in range
    mask_m = m_offsets < n_index
    mask_n = n_offsets < n_other

    # Load the index values
    m_indices = tl.load(index_ptr + m_offsets, mask=mask_m, other=0)

    # Expand for broadcasting
    m_indices = tl.broadcast_to(m_indices[:, None], [BLOCK_M, BLOCK_N])
    mask = mask_m[:, None] & mask_n[None, :]

    # Compute the location to fill
    if dim == 0:
        # If dim=0, each index refers to a row
        row = m_indices
        col = n_offsets[None, :]
    else:
        # If dim=1, each index refers to a column
        row = n_offsets[None, :]
        col = m_indices

    # Stride-based addresses
    addr = row * stride0 + col * stride1
    tl.store(data_ptr + addr, value, mask=mask)


def index_fill_(data: torch.Tensor, dim: int, index: torch.Tensor, value: float) -> torch.Tensor:
    """
    In-place version of index_fill that fills elements of 'data' with 'value'
    along dimension 'dim' using indices from 'index'.
    """
    # Only supports 2D tensors for demonstration.
    assert data.dim() == 2, "Only 2D tensors are supported in this example."
    assert dim in (0, 1), "dim must be 0 or 1 for this example."

    # Prepare parameters
    n_index = index.shape[0]
    n_other = data.shape[1 - dim]
    grid = lambda META: (
        (n_index + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (n_other + META['BLOCK_N'] - 1) // META['BLOCK_N'],
    )

    _index_fill_kernel_2d[grid](
        data,
        index,
        value,
        data.stride(0),
        data.stride(1),
        n_index,
        n_other,
        dim,
        BLOCK_M=32,
        BLOCK_N=32
    )
    return data
