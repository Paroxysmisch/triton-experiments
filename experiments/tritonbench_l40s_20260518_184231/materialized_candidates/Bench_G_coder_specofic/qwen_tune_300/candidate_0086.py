import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    index,
    grad_output,
    grad_source,
    stride_source_n,
    stride_source_c,
    stride_grad_output_n,
    stride_grad_output_c,
    stride_grad_source_n,
    stride_grad_source_c,
    N: tl.constexpr,
    C: tl.constexpr,
    INDEX_SELECT_CAT_BWD_BLOCK_SIZE_INDEX: tl.constexpr,
    INDEX_SELECT_CAT_BWD_BLOCK_SIZE_COL: tl.constexpr,
):
    # Get the batch index
    batch_index = tl.program_id(0)

    # Compute the offset for the batch
    grad_output_batch_offset = batch_index * stride_grad_output_n
    grad_source_batch_offset = batch_index * stride_grad_source_n

    # Define the range of indices for this program instance
    index_range_start = batch_index * INDEX_SELECT_CAT_BWD_BLOCK_SIZE_INDEX
    index_range_end = index_range_start + INDEX_SELECT_CAT_BWD_BLOCK_SIZE_INDEX

    # Iterate over the index range
    for index_ in range(index_range_start, index_range_end):
        # Compute the offset for the current index
        index_offset = index_ * stride_source_n
        n_mask = index_offset < N * stride_source_n

        # Load the index value
        index_val = tl.load(index + index_offset, mask=n_mask)

        # Compute the offset for the current index value
        grad_output_index_offset = index_val * stride_grad_output_c
        grad_source_index_offset = index_val * stride_grad_source_c

        # Load the gradient output
        grad_output_val = tl.load(
            grad_output + grad_output_batch_offset + grad_output_index_offset,
            mask=n_mask,
        )

        # Store the gradient output in the gradient source
        tl.store(
            grad_source + grad_source_batch_offset + grad_source_index_offset,
            grad_output_val,
            mask=n_mask,
        )
    return


def index_select_cat_bwd(
    index: torch.Tensor, grad_output: torch.Tensor, N: int, C: int
) -> torch.Tensor:
    # Ensure index is a 2D tensor on CUDA
    assert index.ndim == 2 and index.is_cuda
    # Ensure grad_output is a 2D tensor on CUDA
    assert grad_output.ndim == 2 and grad_output.is_cuda
    # Ensure the number of columns in index matches the stride of grad_output
    assert index.shape[1] == grad_output.stride(0)
    # Ensure grad_source is newly created with the appropriate shape and strides
    grad_source = torch.empty_like(index, dtype=torch.float32, device="cuda")
    assert grad_source.stride() == grad_output.stride()

    # Define the grid size for the kernel launch
    grid = lambda meta: (
        triton.cdiv(index.shape[0], meta["INDEX_SELECT_CAT_BWD_BLOCK_SIZE_INDEX"]),
    )

    # Launch the Triton kernel
    index_select_cat_bwd_kernel[grid](
        index,
        grad_output,
        grad_source,
        index.stride(0),
        index.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_source.stride(0),
        grad_source.stride(1),
        N,
        C,
        # Set block sizes as constants
        INDEX_SELECT_CAT_BWD_BLOCK_SIZE_INDEX=128,
        INDEX_SELECT_CAT_BWD_BLOCK_SIZE_COL=128,
    )
    return grad_source
