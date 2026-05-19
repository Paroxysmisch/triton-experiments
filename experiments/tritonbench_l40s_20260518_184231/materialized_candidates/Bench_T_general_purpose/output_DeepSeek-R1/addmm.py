import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    # Pointers to matrices
    mat1_ptr, mat2_ptr, input_ptr, out_ptr,
    # Matrix dimensions
    n, m, p,
    # Strides
    stride_m1_row, stride_m1_col,
    stride_m2_row, stride_m2_col,
    stride_in_row, stride_in_col,
    stride_out_row, stride_out_col,
    # Meta-parameters
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_TYPE: tl.constexpr,  # Accumulator type (tl.float32 or tl.float16)
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    row_start = pid0 * BLOCK_SIZE
    col_start = pid1 * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks to handle boundary conditions
    row_mask = row_offsets < n
    col_mask = col_offsets < p
    block_mask = row_mask[:, None] & col_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=ACC_TYPE)
    
    # Compute block-wise matrix multiplication
    for k in range(0, m, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < m
        
        # Load mat1 block
        a_ptrs = mat1_ptr + row_offsets[:, None] * stride_m1_row + k_offsets[None, :] * stride_m1_col
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load mat2 block
        b_ptrs = mat2_ptr + k_offsets[:, None] * stride_m2_row + col_offsets[None, :] * stride_m2_col
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        # Accumulate matrix product
        acc += tl.dot(a, b, out_dtype=ACC_TYPE)
    
    # Scale by alpha
    acc = acc * alpha
    
    # Add beta * input if beta != 0
    if beta != 0.0:
        input_ptrs = input_ptr + row_offsets[:, None] * stride_in_row + col_offsets[None, :] * stride_in_col
        input_block = tl.load(input_ptrs, mask=block_mask, other=0.0)
        acc += beta * input_block
    
    # Store output
    out_ptrs = out_ptr + row_offsets[:, None] * stride_out_row + col_offsets[None, :] * stride_out_col
    tl.store(out_ptrs, acc, mask=block_mask)

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Dispatch to PyTorch for sparse tensors
    if input.is_sparse or mat1.is_sparse or mat2.is_sparse:
        if out is not None:
            return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
        else:
            return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    
    # Validate mat1 and mat2 dimensions
    if mat1.dim() != 2:
        raise RuntimeError("mat1 must be a 2D tensor")
    if mat2.dim() != 2:
        raise RuntimeError("mat2 must be a 2D tensor")
    n, m = mat1.shape
    m2, p = mat2.shape
    if m != m2:
        raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({n}x{m} and {m2}x{p})")
    
    # Broadcast input to (n, p)
    try:
        input_expanded = input.broadcast_to((n, p))
    except RuntimeError:
        raise RuntimeError(f"input shape {input.shape} is not broadcastable to ({n}, {p})")
    
    # Ensure consistent device and dtype
    device = input.device
    assert device == mat1.device == mat2.device, "All tensors must be on the same device"
    dtype = input.dtype
    assert dtype == mat1.dtype == mat2.dtype, "All tensors must have the same dtype"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((n, p), dtype=dtype, device=device)
    else:
        if out.shape != (n, p):
            raise RuntimeError(f"out tensor shape {out.shape} does not match expected shape ({n}, {p})")
        if out.dtype != dtype:
            raise RuntimeError(f"out tensor dtype {out.dtype} does not match input dtype {dtype}")
    
    # Define block size and compute grid
    BLOCK_SIZE = 64  # Tune based on hardware
    grid = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(p, BLOCK_SIZE))
    
    # Determine accumulator type
    ACC_TYPE = tl.float32 if dtype in [torch.float32, torch.bfloat16, torch.float16] else dtype
    
    # Launch kernel
    addmm_kernel[grid](
        mat1, mat2, input_expanded, out,
        n, m, p,
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        input_expanded.stride(0), input_expanded.stride(1),
        out.stride(0), out.stride(1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
        ACC_TYPE=ACC_TYPE,
    )
    
    return out
