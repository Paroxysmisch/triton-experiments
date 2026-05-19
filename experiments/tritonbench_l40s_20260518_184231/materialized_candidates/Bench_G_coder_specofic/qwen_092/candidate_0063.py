import torch

def uint_to_uniform_float(x: torch.Tensor) -> torch.Tensor:
    return x / (2**32 - 1)

def uniform_(out_ptr: torch.Tensor, N: int, philox_seed: int, philox_offset: int, from_: float, to: float):
    block_size = 256
    num_warps = 4
    grid_size = (N + block_size * 4 - 1) // (block_size * 4)

    with torch.cuda.device(out_ptr.device):
        uniform_kernel[grid_size, block_size, num_warps](out_ptr, N, philox_seed, philox_offset, from_, to)

# Example usage
N = 1024
out_ptr = torch.empty(N, dtype=torch.float32, device='cuda')
philox_seed = 12345
philox_offset = 0
from_ = 0.0
to = 1.0

uniform_(out_ptr, N, philox_seed, philox_offset, from_, to)
print(out_ptr)
