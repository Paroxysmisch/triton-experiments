triton
import triton
import triton.language as tl

@triton.jit
def fused_pairwise_distance_adaptive_avg_pool2d_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    x1_shape,
    x2_shape,
    output_size,
    p,
    eps,
    n_elements,
    block_size_x,
    block_size_y,
    grid_size_x,
    grid_size_y,
):
    # Calculate indices for the current thread
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Calculate the total number of elements in the input tensors
    n_rows = x1_shape[0]
    n_cols = x1_shape[1]

    # Loop through all elements in the input tensors
    for i in range(row * block_size_y, min(n_rows, (row + 1) * block_size_y)):
        for j in range(col * block_size_x, min(n_cols, (col + 1) * block_size_x)):
            # Calculate the index in the input tensors
            idx1 = i * n_cols + j
            idx2 = idx1

            # Apply adaptive average pooling
            pool_i = i // output_size
            pool_j = j // output_size
            pool_idx1 = pool_i * output_size + pool_j
            pool_idx2 = pool_idx1

            # Load values from the input tensors
            x1_val = tl.load(x1_ptr + idx1)
            x2_val = tl.load(x2_ptr + idx2)

            # Calculate the difference and apply the norm
            diff = x1_val - x2_val
            dist = tl.math.pow(tl.abs(diff), p)
            dist += eps
            dist = tl.math.pow(dist, 1 / p)

            # Store the result in the output tensor
            out_ptr[idx1] = dist


# Triton wrapper function
def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Ensure input tensors are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    # Get the shape of the input tensors
    x1_shape = x1.shape
    x2_shape = x2.shape

    # Check if the shapes are compatible for pooling
    assert x1_shape == x2_shape, "Input tensors must have the same shape"

    # Create an output tensor
    if keepdim:
        out_shape = list(x1_shape)
        out_shape[-2:] = [output_size, output_size]
    else:
        out_shape = [output_size, output_size]

    out = torch.empty(out_shape, dtype=x1.dtype, device=x1.device)

    # Set up Triton kernel parameters
    block_size_x = 32
    block_size_y = 32
    grid_size_x = (out_shape[0] + block_size_x - 1) // block_size_x
    grid_size_y = (out_shape[1] + block_size_y - 1) // block_size_y

    # Launch the Triton kernel
    fused_pairwise_distance_adaptive_avg_pool2d_kernel[
        grid_size_x, grid_size_y, (block_size_x, block_size_y, 1)
    ](
        x1.data_ptr(),
        x2.data_ptr(),
        out.data_ptr(),
        x1_shape,
        x2_shape,
        output_size,
        p,
        eps,
        out.numel(),
        block_size_x,
        block_size_y,
        grid_size_x,
        grid_size_y,
    )

    return out
