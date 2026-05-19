import torch
import triton
import triton.language as tl

@triton.jit
def nested3(
    in_ptr, out_ptr,
    stride_m, stride_n,
    n_rows, n_cols,
    BLOCK_COLS: tl.constexpr
):
    pid = tl.program_id(0)
    start_col = pid * BLOCK_COLS
    # Iterate over rows in steps of 2
    for row in range(0, n_rows, 2):
        # Three nested loops over i, j, k
        for i in range(2):  # Row index within the 2x2 tile
            for k in range(2):  # Which 2x2 tile within the 4-column block (0 or 1)
                for j in range(2):  # Column index within the 2x2 tile
                    col = start_col + k * 2 + j
                    current_row = row + i
                    # Bounds checking
                    if current_row < n_rows and col < n_cols:
                        offset = current_row * stride_m + col * stride_n
                        a_ptr = in_ptr + offset
                        c_ptr = out_ptr + offset
                        val = tl.load(a_ptr)
                        tl.store(c_ptr, val)

def wrapper_nested3(x: torch.Tensor):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_COLS = 4  # Each block processes 4 columns
    grid = (triton.cdiv(n_cols, BLOCK_COLS),)
    nested3[grid](
        x, output,
        x.stride(0), x.stride(1),
        n_rows, n_cols,
        BLOCK_COLS=BLOCK_COLS
    )
    print("Output tensor:")
    print(output)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    n_rows, n_cols = 8, 16
    x = torch.arange(n_rows * n_cols, dtype=torch.int32, device='cuda').reshape(n_rows, n_cols)
    print("Input tensor:")
    print(x)
    wrapper_nested3(x)
