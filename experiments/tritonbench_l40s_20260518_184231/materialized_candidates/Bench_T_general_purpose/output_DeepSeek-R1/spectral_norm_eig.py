import torch
import triton
import triton.language as tl

@triton.jit
def max_abs_eigenvalue_kernel(eigenvalues_ptr, output_ptr, n_eigenvalues,
                              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * n_eigenvalues
    max_abs = 0.0
    for offset in range(0, n_eigenvalues, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_eigenvalues
        eigenvalues = tl.load(eigenvalues_ptr + row_start + offsets, mask=mask, other=0.0)
        abs_values = tl.abs(eigenvalues)
        current_max = tl.max(tl.where(mask, abs_values, 0.0), axis=0)
        max_abs = tl.maximum(max_abs, current_max)
    tl.store(output_ptr + pid, max_abs)

def spectral_norm_eig(A: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert A.shape[-1] == A.shape[-2], "Input must be a square matrix."
    
    eigenvalues = torch.linalg.eigvals(A)  # Shape (*, n)
    n_eigenvalues = eigenvalues.shape[-1]
    
    original_shape = eigenvalues.shape
    eigenvalues_flat = eigenvalues.view(-1, n_eigenvalues)
    num_matrices = eigenvalues_flat.shape[0]
    
    output_dtype = eigenvalues.real.dtype  # Get the corresponding real dtype
    output_flat = torch.empty(num_matrices, dtype=output_dtype, device=eigenvalues.device)
    
    # Choose a block size that is a power of two and <= 1024
    BLOCK_SIZE = triton.next_power_of_2(n_eigenvalues)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (num_matrices,)
    max_abs_eigenvalue_kernel[grid](eigenvalues_flat, output_flat, n_eigenvalues, BLOCK_SIZE=BLOCK_SIZE)
    
    output = output_flat.view(original_shape[:-1])
    
    if out is not None:
        out.copy_(output)
        return out
    return output
