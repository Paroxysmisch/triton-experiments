import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A_ptr, P_ptr, L_ptr, U_ptr, N, BLOCK_SIZE: tl.constexpr, pivot: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Initialize permutation matrix P
    if row == col:
        tl.store(P_ptr + row * N + col, 1.0, mask=row == col)

    A = tl.load(A_ptr + row * N + col, mask=row >= col)
    P = tl.load(P_ptr + row * N + col, mask=row >= col)
    L = tl.zeros((N, N), dtype=tl.float32)
    U = tl.zeros((N, N), dtype=tl.float32)

    for k in range(row, N):
        if pivot:
            max_row = k
            max_val = abs(A[k, k])
            for i in range(k+1, N):
                val = abs(A[i, k])
                if val > max_val:
                    max_row = i
                    max_val = val

            if max_row != k:
                # Swap rows in A and P
                for j in range(N):
                    temp = A[max_row, j]
                    A[max_row, j] = A[k, j]
                    A[k, j] = temp

                    temp = P[max_row, j]
                    P[max_row, j] = P[k, j]
                    P[k, j] = temp

        if k == row:
            U[row, row] = A[row, row]
        else:
            U[row, k] = A[row, k] / U[k, k]

        for j in range(row, N):
            if j < k:
                L[j, row] = A[j, row]
            else:
                L[j, row] = A[j, row] - U[k, row] * L[j, k]

        for j in range(k, N):
            A[row, j] = A[row, j] - U[k, row] * L[row, j]

    # Store results
    tl.store(L_ptr + row * N + col, L[row, col], mask=row >= col)
    tl.store(U_ptr + row * N + col, U[row, col], mask=row >= col)
