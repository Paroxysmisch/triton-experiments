import triton
import triton.language as tl
import torch

# -------------------
# Triton kernel: computes the maximum of absolute values in a 1D array
# -------------------
@triton.jit
def _abs_max_kernel(
    ptr_in, ptr_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data, compute absolute value, and initialize a local max
    data = tl.load(ptr_in + offsets, mask=mask, other=0)
    local_max = tl.abs(data)

    # Reduce to find block-wide maximum
    # (This reduction uses a tree-based approach for a 1D block)
    half = BLOCK_SIZE // 2
    while half > 0:
        tmp = tl.select(offsets + half < n_elements, tl.roll(local_max, shifts=half, dims=0), 0.0)
        local_max = tl.maximum(local_max, tmp)
        half //= 2

    # Store the block result (only lane 0 holds the final max for this block)
    if tl.static_any(tl.arange(0, BLOCK_SIZE) == 0):
        out_idx = pid
        tl.store(ptr_out + out_idx, local_max[0])

def spectral_norm_eig(A: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    spectral_norm_eig(A, *, out=None) -> Tensor

    Computes the spectral norm (the largest absolute eigenvalue) of a square matrix using its eigenvalues.
    Supports float, double, cfloat, and cdouble dtypes, and handles batches of square matrices.

    A (Tensor): Shape (..., n, n), where ... is zero or more batch dimensions.
    out (Tensor, optional): Output tensor. Ignored if None.
    """

    # Compute eigenvalues
    # For real/complex inputs, torch.linalg.eigvals handles float/double/cfloat/cdouble
    eigvals = torch.linalg.eigvals(A)  # shape (..., n)

    # Flatten batch for simpler processing
    batch_shape = eigvals.shape[:-1]
    n = eigvals.shape[-1]
    eigvals_2d = eigvals.reshape(-1, n)  # shape (prod_batch, n)

    # Prepare output array for block-wise reductions
    device = eigvals.device
    num_batches = eigvals_2d.shape[0]
    # We'll handle partial block reductions, collecting intermediate maxima here
    partial_max = torch.empty(num_batches, dtype=eigvals.real.dtype, device=device)

    # Launch the kernel for each batch along dimension 0
    # Each batch: we compute the maximum absolute eigenvalue with a block-level reduction
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Run a loop over each batch item (since each might require its own kernel call)
    # for large batch sizes you may prefer a fused approach, but for clarity we keep it simple here.
    for i in range(num_batches):
        x_i = eigvals_2d[i]
        # Make sure the input is float or cfloat on the device for kernel usage
        # We'll handle complex as 2 floats if the kernel used direct memory, but here we just store abs beforehand
        # to keep it simple: we store abs values into a float buffer.
        x_i_abs = torch.abs(x_i).float()  # ensure float for kernel
        buf_in = x_i_abs.contiguous()
        result_buffer = torch.empty(((n + BLOCK_SIZE - 1) // BLOCK_SIZE,), device=device, dtype=buf_in.dtype)

        _abs_max_kernel[grid](
            buf_in,                   # ptr_in
            result_buffer,            # ptr_out
            n,
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Now reduce the partial results on the CPU or GPU
        partial_max[i] = result_buffer.max()

    # Now partial_max holds the maximum absolute eigenvalue for every batch
    spectral_vals = partial_max.reshape(batch_shape)

    # If out is provided, write the result there
    if out is not None:
        out.copy_(spectral_vals)
        return out
    return spectral_vals
