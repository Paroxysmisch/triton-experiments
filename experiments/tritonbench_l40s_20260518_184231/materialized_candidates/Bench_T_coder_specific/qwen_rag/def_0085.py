import triton
import triton.language as tl

@triton.jit
def matrix_power_eig_kernel(output_ptr, A_ptr, U_ptr, D_ptr, inv_U_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_offset = pid * BLOCK_SIZE
    mask = block_offset < n

    # Load A[i, j]
    i, j = tl.meshgrid(tl.arange(BLOCK_SIZE), tl.arange(BLOCK_SIZE))
    a_ij = tl.load(A_ptr + i * n + j, mask=mask)

    # Compute V @ D @ V^-1
    u_ji = tl.load(U_ptr + j * n + i, mask=mask)
    d_i = tl.load(D_ptr + i, mask=mask)
    inv_u_ji = tl.load(inv_U_ptr + j * n + i, mask=mask)

    # Initialize output
    output = tl.zeros((BLOCK_SIZE,), dtype=a_ij.dtype)

    # Perform the matrix multiplication
    for m in range(n):
        v_mj = tl.load(U_ptr + m * n + j, mask=mask)
        d_mm = tl.load(D_ptr + m, mask=mask)
        inv_v_mi = tl.load(inv_U_ptr + m * n + i, mask=mask)
        output += u_ji * d_mm * inv_u_ji * v_mj

    # Store the result
    tl.store(output_ptr + i * n + j, output, mask=mask)
