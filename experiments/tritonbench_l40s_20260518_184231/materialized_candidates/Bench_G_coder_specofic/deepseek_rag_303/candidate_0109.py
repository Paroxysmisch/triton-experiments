import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    input_ptr, output_ptr, M, N, TILE_N: tl.constexpr,
):
    # Kernel parameters
    pid = tl.program_id(0)

    # rows tile will be computed by this program
    num_tiles = tl.cdiv(M, TILE_N)
    pidx = pid % num_tiles
    pidy = pid // num_tiles

    # each program will process TILE_N rows and N columns
    rows_offsets = pidy * TILE_N + tl.arange(0, TILE_N)
    cols_offsets = pidx * TILE_N + tl.arange(0, TILE_N)

    input_ptrs = input_ptr + (rows_offsets[:, None] * N + cols_offsets[None, :])

    # load input data in a tiling pattern
    input_tile = tl.load(input_ptrs, mask=(rows_offsets[:, None] < M) & (cols_offsets[None, :] < N), other=-float("inf"))

    # Step 1: Compute max
    max_row = tl.max(input_tile, axis=1)

    # Step 2: Substract max and compute exp(x)
    input_minus_max = input_tile - max_row[:, None]
    exp_tile = tl.exp(input_minus_max)

    # Step 3: Compute sum
    sum_row = tl.sum(exp_tile, axis=1)

    # Step 4: Divide
    softmax_tile = exp_tile / sum_row[:, None]

    # Compute output pointer
    output_ptrs = output_ptr + (rows_offsets[:, None] * N + cols_offsets[None, :])

    # Store data in output
    tl.store(output_ptrs, softmax_tile, mask=(rows_offsets[:, None] < M) & (cols_offsets[None, :] < N))

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: The softmax of x.
    """
    M, N = x.shape

    # TILE_N is the size of the tile, which is LCM of {1, 2, 4, 8, 16}
    TILE_N = 16

    # Make sure the size across the dimension is a multiple of TILE_N
    if M % TILE_N != 0:
        M = triton.next_power_of_2(M)
        x = x.view(x.shape + torch.Size([1]))

    if N % TILE_N != 0:
        N = triton.next_power_of_2(N)
        x = x.view(torch.Size([1]) + x.shape)

    # Define helper lambda function
    num_entities = lambda meta: (triton.cdiv(M, meta["TILE_N"]) * triton.cdiv(N, meta["TILE_N"]),)

    # placeholder for the result
    if x.is_cuda:
        output = torch.empty((M, N), dtype=torch.float32, device=x.device)
    else:
        output = torch.empty((M, N), dtype=torch.float32)
 
    # Dispatch Triton kernel
    softmax_kernel_online_v2[num_entities](x, output, M, N, TILE_N=TILE_N)

    return output
