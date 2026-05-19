import torch
import triton
import triton.language as tl

# Triton kernel function
@triton.jit
def _quantize_rowwise(
    input_ptr: tl.tensor, 
    output_ptr: tl.tensor, 
    max_values_ptr: tl.tensor, 
    n_rows: tl.int32, 
    n_cols: tl.int32, 
    BLOCK_SIZE: tl.constexpr, 
    P2: tl.constexpr
):
    row = tl.program_id(0)
    if row < n_rows:
        max_val = 0.0
        for col in range(0, n_cols, BLOCK_SIZE):
            offsets = row * n_cols + col + tl.arange(0, BLOCK_SIZE)
            input_vals = tl.load(input_ptr + offsets, mask=offsets < row * n_cols + n_cols)
            max_val = tl.max(max_val, tl.max(tl.abs(input_vals), axis=0))

        tl.store(max_values_ptr + row, max_val)

        for col in range(0, n_cols, BLOCK_SIZE):
            offsets = row * n_cols + col + tl.arange(0, BLOCK_SIZE)
            input_vals = tl.load(input_ptr + offsets, mask=offsets < row * n_cols + n_cols)
            quantized_vals = (127.0 * input_vals / max_val).to(tl.int8)
            tl.store(output_ptr + offsets, quantized_vals, mask=offsets < row * n_cols + n_cols)

# Main function to prepare tensors and launch the kernel
def quantize_rowwise(input_tensor: torch.Tensor):
    assert input_tensor.dim() == 2, "Input tensor must be 2D"
    n_rows, n_cols = input_tensor.shape

    # Allocate output tensors
    output_tensor = torch.empty((n_rows, n_cols), dtype=torch.int8, device=input_tensor.device)
    max_values_tensor = torch.empty((n_rows,), dtype=torch.float32, device=input_tensor.device)

    # Define grid and block sizes
    BLOCK_SIZE = 128
    P2 = 128
    grid = (n_rows,)

    # Launch the kernel
    _quantize_rowwise[grid](
        input_tensor, 
        output_tensor, 
        max_values_tensor, 
        n_rows, 
        n_cols, 
        BLOCK_SIZE, 
        P2
    )

    return output_tensor, max_values_tensor

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    input_tensor = torch.randn((1024, 1024), dtype=torch.float32, device="cuda")

    # Perform row-wise quantization
    output_tensor, max_values_tensor = quantize_rowwise(input_tensor)

    print("Quantized Tensor:", output_tensor)
    print("Max Values Tensor:", max_values_tensor)
