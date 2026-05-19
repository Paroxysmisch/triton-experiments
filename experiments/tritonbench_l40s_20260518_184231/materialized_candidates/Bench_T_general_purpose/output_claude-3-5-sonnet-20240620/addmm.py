import triton
import triton.language as tl

@triton.jit
def addmm_kernel(input_ptr, mat1_ptr, mat2_ptr, out_ptr, beta, alpha, n, m, p):
    # Define the grid size
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Check bounds
    if row >= n or col >= p:
        return

    # Load input, mat1, and mat2
    input_val = tl.load(input_ptr + row * p + col) if beta != 0 else 0
    mat1_val = tl.load(mat1_ptr + row * m + tl.arange(0, m))
    mat2_val = tl.load(mat2_ptr + tl.arange(0, m) * p + col)

    # Compute the matrix multiplication
    result = tl.dot(mat1_val, mat2_val)

    # Add the input scaled by beta and the result scaled by alpha
    out_val = beta * input_val + alpha * result

    # Store the result
    tl.store(out_ptr + row * p + col, out_val)

def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    # Ensure input shapes are compatible
    n, m = mat1.shape
    _, p = mat2.shape

    # Create output tensor if not provided
    if out is None:
        out = torch.empty((n, p), dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid = (n, p)
    addmm_kernel[grid](input, mat1, mat2, out, beta, alpha, n, m, p)

    return out
