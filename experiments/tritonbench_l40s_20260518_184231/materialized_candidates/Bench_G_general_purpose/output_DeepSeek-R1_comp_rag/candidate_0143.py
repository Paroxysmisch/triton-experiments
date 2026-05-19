import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n, N_COLS: tl.constexpr):
    pid = tl.program_id(0)
    start_col = pid * 4  # Each kernel handles 4 columns

    offs_m = tl.arange(0, 2)  # 2 rows
    offs_n = tl.arange(0, 2)  # 2 columns

    # Base input pointers for the 2x2 tile
    a_ptrs = in_ptr + (offs_m[:, None] * stride_m + (start_col + offs_n[None, :]) * stride_n)
    a = tl.load(a_ptrs)

    # Initialize output pointers for the starting 2x2 tile
    c_ptrs = out_ptr + (offs_m[:, None] * stride_m + (start_col + offs_n[None, :]) * stride_n)

    # Nested loops over i, j, k
    for i in range(0, 2):
        a1 = a  # Load the initial 2x2 tile

        for j in range(0, 2):
            # Move to next 2 columns
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k in range(0, 2):
                # Move to next 2 columns
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)

                # Store a1, a2, a3 to output
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n

            # Increment after j loop
            a_ptrs += 2 * stride_n

def wrapper_nested3():
    n_rows = 4
    n_cols = 48
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape(n_rows, n_cols)
    output = torch.zeros_like(x)

    grid = (n_cols // 4,)  # Each kernel instance processes 4 columns

    nested3[grid](x, output, x.stride(0), x.stride(1), n_cols)
    print("Output tensor:")
    print(output)

# Execute the wrapper
if __name__ == "__main__":
    wrapper_nested3()
