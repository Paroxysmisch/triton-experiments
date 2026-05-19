import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    x_ptr,
    y_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    # The program id is the row of the tile we are computing.
    pid = tl.program_id(axis=0)
    # The stride represents how much we need to increase the pointer to advance 1 row.
    # It's also the amount we need to increase the pointer when moving an element across
    # one row.
    stride_amnt_row = tl.num_programs(axis=0) * TILE_N
    # row_start is the starting pointer for the current row we are computing.
    row_start = x_ptr + (pid * TILE_N) * N + tl.arange(0, TILE_N)
    # col_offsets are the offsets into the next column for each element in the row.
    col_offsets = tl.arange(0, TILE_N)
    # Load the row into SRAM, using a pointer structure that allows for easy indexing
    # into the row.
    row_ram = tl.load(row_start + col_offsets * N, mask=col_offsets < M, other=-float("inf"))
    # The maximum value in the row.
    row_max = tl.max(row_ram)
    # The row minus the maximum value, minus the maximum value.
    row_minus_max = row_ram - row_max
    # Exponentiated values.
    exp_row = tl.exp(row_minus_max)
    # The sum of the exponentiated values.
    sum_row = tl.sum(exp_row)
    # Normalization constant.
    normalization_const = tl.exp(row_max) + sum_row
    # Normalized output.
    output_row = exp_row / normalization_const
    # Write back output to DRAM.
    y_ptr_row_start = y_ptr + (pid * TILE_N) * N + col_offsets
    tl.store(y_ptr_row_start, output_row, mask=col_offsets < M)

def softmax(x):
    M, N = x.shape
    # The number of tiles we need to cover the matrix is dictated by the number of
    # rows. We'll use a tiling approach to compute the softmax, using a tiling
    # schedule that tiles the rows of the matrix.
    # The tiling schedule is designed to balance the computational complexity and
    # memory usage.
    num_tiles = M
    # The output placeholder is initialized with the softmax of the input.
    y = torch.empty_like(x)
    # We'll use a grid-stride loop to compute the softmax. The grid stride loop
    # iterates over the rows of the matrix, and for each row, it uses a Triton kernel
    # to compute the softmax.
    # The kernel expects a contiguous chunk of memory and we cannot use stride
    # so we need to call it multiple times.
    tile_sizes = [1]
    for _ in range(num_tiles):
        tile_sizes.append(tile_sizes[-1] * 2)
    tile_sizes.pop(0)
    for i in range(num_tiles):
        tile_size = tile_sizes[i]
        # Launch the Triton kernel.
        grid = (triton.cdiv(M, tile_size),)
        softmax_kernel_online_v2[grid](
            x,
            y,
            M,
            N,
            TILE_N=tile_size,
        )
    return y

x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32)
y = softmax(x)
print(y)
