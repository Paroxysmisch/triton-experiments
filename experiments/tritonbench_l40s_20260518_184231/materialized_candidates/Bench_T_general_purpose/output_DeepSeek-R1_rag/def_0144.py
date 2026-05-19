import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_kernel(
    product_ptr,
    c_ptr,
    alpha,
    beta,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load product and C values
    prod_val = tl.load(product_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    # Compute new value: alpha * product + beta * C
    new_val = alpha * prod_val + beta * c_val
    # Store back into C
    tl.store(c_ptr + offsets, new_val, mask=mask)

@triton.jit
def row_dot_kernel(
    c_ptr,
    p_cols,
    result_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p_cols
    # Load elements from row 0 and 1
    val0 = tl.load(c_ptr + 0 * p_cols + offsets, mask=mask)
    val1 = tl.load(c_ptr + 1 * p_cols + offsets, mask=mask)
    # Compute products
    products = val0 * val1
    # Sum the products in this block
    sum_partial = tl.sum(products, axis=0)
    # Atomically add to the result
    tl.atomic_add(result_ptr, sum_partial)

def matrix_multiply_and_row_dot(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    beta: float,
    C: torch.Tensor
) -> torch.Tensor:
    # Compute product = A @ B using PyTorch's optimized mm
    product = torch.mm(A, B)
    product = product.contiguous()
    
    # Handle C's contiguity and potential copy back to original
    original_C = C
    if not original_C.is_contiguous():
        C = original_C.contiguous()
    
    n, p = C.shape
    n_elements = n * p
    
    # Launch scaled_add_kernel to compute C = alpha * product + beta * C
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scaled_add_kernel[grid](product, C, alpha, beta, n_elements, BLOCK_SIZE=1024)
    
    # Copy back to original C if necessary
    if original_C is not C:
        original_C.copy_(C)
        C = original_C  # Ensure C refers to the original tensor for dot product
    
    # Check if C has at least two rows
    if C.shape[0] < 2:
        raise ValueError("C must have at least two rows for dot product computation.")
    
    # Create result tensor on the same device as C
    result = torch.zeros(1, device=C.device, dtype=C.dtype)
    p_cols = p
    
    # Launch row_dot_kernel to compute the dot product of the first two rows
    grid_dot = lambda meta: (triton.cdiv(p_cols, meta['BLOCK_SIZE']),)
    row_dot_kernel[grid_dot](C, p_cols, result, BLOCK_SIZE=1024)
    
    return result.squeeze()
