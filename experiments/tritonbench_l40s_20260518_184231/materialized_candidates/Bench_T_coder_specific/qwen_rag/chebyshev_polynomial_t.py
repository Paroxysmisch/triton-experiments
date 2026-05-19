import triton
import triton.language as tl

@triton.jit
def chebyshev_polynomial_t_kernel(
    X_ptr,
    N_ptr,
    Out_ptr,
    X_shape,
    N_shape,
    stride_X_0,
    stride_X_1,
    stride_N_0,
    stride_N_1,
    stride_Out_0,
    stride_Out_1,
    num_elements,
):
    # Compute thread index
    pid = tl.program_id(0)
    grid_size = tl.cdiv(num_elements, BLOCK_SIZE)

    # Initialize pointers
    X_base = X_ptr + pid * stride_X_0
    N_base = N_ptr + pid * stride_N_0
    Out_base = Out_ptr + pid * stride_Out_0

    # Load input values
    x = tl.load(X_base)
    n = tl.load(N_base)

    # Initialize T_{n-1} and T_n
    T_nm2 = tl.float32(1.0)
    T_n = x if n == 1 else tl.float32(0.0)

    # Compute T_n using the recursive formula
    for _ in range(n):
        T_np1 = 2 * x * T_n - T_nm2
        T_nm2 = T_n
        T_n = T_np1

    # Store the result
    tl.store(Out_base, T_n)
