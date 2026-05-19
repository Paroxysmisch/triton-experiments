import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(x_ptr, y_ptr, z_ptr,
                             x_num_rows, y_num_rows,
                             x_num_cols, y_num_cols,
                             p_norm, eps_norm, eps_distance,
                             BLOCK_SIZE_X, BLOCK_SIZE_Y):
    # Define the grid and block dimensions
    grid = tl.grid(x_num_rows, y_num_cols, BLOCK_SIZE_X, BLOCK_SIZE_Y)
    block = tl.block(BLOCK_SIZE_X, BLOCK_SIZE_Y)

    # Load the input tensors
    x = tl.load(x_ptr + block.id[0] * x_num_cols + block.id[1] * x_num_cols * x_num_rows)
    y = tl.load(y_ptr + block.id[0] * y_num_cols + block.id[1] * y_num_cols * y_num_rows)

    # Compute the pairwise distance
    z = tl.sum(tl.pow(tl.abs(x - y), p_norm))

    # Normalize the distance
    z_norm = z / (tl.sum(tl.pow(tl.abs(x), p_norm)) + eps_norm)

    # Store the result
    tl.store(z_ptr + block.id[0] * y_num_cols + block.id[1] * y_num_cols * x_num_rows, z_norm + eps_distance)

def fused_pairwise_distance_normalize(x1, x2, p_norm=2.0, eps_norm=1e-12, eps_distance=1e-6, keepdim=False):
    # Compute the shape of the input tensors
    x1_shape = x1.shape
    x2_shape = x2.shape

    # Check if the tensors are compatible for pairwise distance calculation
    assert x1_shape == x2_shape, "Input tensors must have the same shape for pairwise distance calculation"

    # Allocate memory for the output tensor
    z = torch.empty(x1_shape, device=x1.device)

    # Call the Triton kernel
    pairwise_distance_kernel[(x1_shape[0], x1_shape[1], 16, 16)](x1.data.ptr, x2.data.ptr, z.data.ptr,
                                                                x1_shape[0], x2_shape[0], x1_shape[1], x2_shape[1],
                                                                p_norm, eps_norm, eps_distance)

    return z
