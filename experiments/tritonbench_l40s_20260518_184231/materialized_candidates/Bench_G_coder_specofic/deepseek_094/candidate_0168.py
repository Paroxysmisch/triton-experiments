import triton
import triton.language as tl

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
])
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.num_programs(axis=0)
    num_pid_n = tl.num_programs(axis=1)

    tile_m = tl.load(A_ptr + pid_m * BLOCK_SIZE_M * K + tl.arange(0, BLOCK_SIZE_K))
    tile_n = tl.load(B_ptr + pid_n * BLOCK_SIZE_N * K + tl.arange(0, BLOCK_SIZE_K))

    C_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_part = tile_m[:, k : k + BLOCK_SIZE_K]
        b_part = tile_n[k : k + BLOCK_SIZE_K, :]
        C_tile += a_part @ b_part

    tl.store(C_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), C_tile)

def triton_matmul(A, B, C):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K, N = B.shape

    grid_m = lambda meta: (meta.size_m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = lambda meta: (meta.size_n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    A_ptr = tl.dereference_bitwise(A.device_buffer_ptr())
    B_ptr = tl.dereference_bitwise(B.device_buffer_ptr())
    C_ptr = tl.dereference_bitwise(C.device_buffer_ptr())

    triton.launch(
        kernel=matmul_kernel,
        grid=(grid_m, grid_n),
        meta={"size_m": M, "size_n": N},
        args=(A_ptr, B_ptr, C_ptr, M, N, K),
        device=A.device,
    )
