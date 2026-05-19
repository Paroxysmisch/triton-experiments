import torch
import triton
import triton.language as tl
from fla.utils import extract_device_arch

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")

if triton.__version__ >= "2.2.0":

    @triton.jit
    def fwd_matmul_tiled(
        index,
        x_ptr,
        w_ptr,
        z_ptr,
        x_row_stride,
        x_inner_stride,
        w_inner_stride,
        w_outer_stride,
        z_row_stride,
        z_inner_stride,
        tiles_per_cta,
        tile_rows,
        tile_inner,
        N,
        K,
        GROUP_SIZE_M: tl.constexpr,
        TILE_ROWS: tl.constexpr,
        TILE_INNER: tl.constexpr,
    ):
        # Map program ids `pid` to the block of C it should compute.
        # We only use 1D blocks here so `pid` is 1D.
        pid = tl.program_id(axis=0)

        # we only handle the case of square matrix mulitplication here
        num_pid_m = tl.cdiv(N, TILE_ROWS)
        num_pid_n = tl.cdiv(K, TILE_INNER)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # create pointers for the first elements of the blocks of C that each program will compute
        offs_m = (pid_m * TILE_ROWS + tl.arange(0, TILE_ROWS)).to(tl.int64)
        offs_n = (pid_n * TILE_INNER + tl.arange(0, TILE_INNER)).to(tl.int64)

        rindex = tl.load(index + offs_m)
        x_block_ptr = x_ptr + (
            rindex[:, None] * x_row_stride + offs_m[:, None] * x_inner_stride
        )
        w_block_ptr = w_ptr + offs_n[None, :] * w_inner_stride
        z_block_ptr = z_ptr + offs_n[None, :] * w_inner_stride

        # Now do the block computation
        accumulator = tl.zeros((TILE_ROWS, TILE_INNER), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, TILE_INNER)):
            x = tl.load(
                x_block_ptr,
                mask=(offs_m[:, None] < N) & (offs_n[None, :] < K - k * TILE_INNER),
                other=0.0,
            )
            w = tl.load(w_block_ptr)
            accumulator += tl.dot(x, w, allow_tf32=True)
            # Advance the ptrs to the next K block.
            x_block_ptr += TILE_INNER * w_inner_stride
            w_block_ptr += TILE_INNER * w_inner_stride
        z = accumulator.to(tl.float32)

        # write back store
        tl.store(z_block_ptr, z, mask=(offs_n[None, :] < K))
else:

    @triton.jit
    def fwd_matmul_tiled(
        x_ptr,
        w_ptr,
        z_ptr,
        x_row_stride,
        x_inner_stride,
        w_inner_stride,
        w_outer_stride,
        z_row_stride,
        z_inner_stride,
        tiles_per_cta,
        tile_rows,
        tile_inner,
        N,
        K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Map program ids `pid` to the block of C it should compute.
        # We only use 1D blocks here so `pid` is 1D.
        pid = tl.program_id(axis=0)

        # we only handle the case of square matrix mulitplication here
        num_pid_n = tl.cdiv(N, tile_rows)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        num_pid_in_group = num_pid_n * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_k = group_id * num_pid_k
        group_size_k = min(num_pid_k - first_pid_k, num_pid_k)
        pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k)
        pid_n = (pid % num_pid_in_group) // group_size_k

        # create pointers for the first elements of the blocks of C that each program will compute
        offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        offs_n = (pid_n * tile_rows + tl.arange(0, tile_rows)).to(tl.int64)

        x_block_ptr = (
            x_ptr
            + (offs_n[:, None] * x_row_stride + offs_k[None, :] * x_inner_stride)
        )
        w_block_ptr = w_ptr + offs_k[:, None] * w_inner_stride
        z_block_ptr = z_ptr + offs_n[:, None] * z_row_stride

        # Now do the block computation
        accumulator = tl.zeros((tile_rows, BLOCK_K), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(
                x_block_ptr,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
            w = tl.load(w_block_ptr)
            accumulator += tl.dot(x, w, allow_tf32=True)
            # Advance the ptrs to the next K block.
            x_block_ptr += BLOCK_K * w_inner_stride
            w_block_ptr += BLOCK_K * w_inner_stride
        z = accumulator.to(tl.float32)

        # write back store
        tl.store(z_block_ptr, z, mask=(offs_k[None, :] < K))

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128, "BLOCK_K":  32}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 128, "BLOCK_K":  64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_N": 128, "BLOCK_K":  32}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K":  64}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_stages=4, num_warps=4),
        ],
        key=["tile_rows", "N", "K"],
    )
    @triton.jit
    def fused_lu_solve(
        A,
        b,
        p,
        n,
        tile_rows,
        TILE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile_idx = tl.program_id(0)
        A = A + tile_idx * tile_rows * n
        b = b + tile_idx * tile_rows
        p = p + tile_idx * tile_rows

        # Load the tile into SRAM.
        # We materialize the entire tile in SRAM even if we don't need all of it.
        tile_mask = (tl.arange(0, TILE_SIZE) < tile_rows)[:, None]

        # Step 1: Apply the permutation.
        p_idx = tl.load(p + tl.arange(0, TILE_SIZE)[:, None], mask=tile_mask, other=0)
        A = A + p_idx * n

        # Step 2: Forward elimination
        upper_mask = (tl.arange(0, TILE_SIZE)[None, :] <= tl.arange(0, TILE_SIZE)[:, None]) & tile_mask & tile_mask.T
        buf_mask = (tl.arange(0, TILE_SIZE)[:, None] < n) & tile_mask
        b_buf = tl.load(b + tl.arange(0, TILE_SIZE)[:, None], mask=buf_mask, other=0).to(tl.float32)
        for i in range(tile_rows):
            A_i = A + i*n + i
            b_i = tl.load(b_buf + i)
            a_ii = tl.load(A_i)
            if a_ii != 0:
                b_buf = tl.where(tl.arange(0, TILE_SIZE)[:, None] > i, b_buf - (tl.load(A + i*n + i+tl.arange(0, TILE_SIZE)[:, None])*b_buf)/a_ii, b_buf)
                tl.store(A_i, 1/a_ii)
            else:
                tl.debug_barrier()
                A_i = A + i*n + i
                a_ii =
