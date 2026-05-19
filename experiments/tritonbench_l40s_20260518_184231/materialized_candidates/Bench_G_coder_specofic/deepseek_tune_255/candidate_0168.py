import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 1, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 4}, num_pid_m=1, num_pid_n=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N
