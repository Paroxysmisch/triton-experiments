import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, 
    x2_ptr, 
    out_ptr,
    N,  # number of rows
    D,  # number of columns
    xnumel,  # total number of elements in the output
    rnumel,  # total number of elements for reduction (D)
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    p_distance: tl.constexpr,
    eps_distance: tl.constexpr
):
    # Global index in terms of output
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # Map the 1D index into (i, j) to compute distance[i,j]
    i = xindex // N
    j = xindex % N

    # Each element in the output accumulates over dimension D
    acc = tl.full([XBLOCK], 0.0, tl.float32)

    # Reduce in chunks of RBLOCK along dimension D
    for rstart in range(0, rnumel, RBLOCK):
        rrange = rstart + tl.arange(0, RBLOCK)
        rmask = rrange < rnumel

        # Compute pointers
        ptr_x1 = x1_ptr + i * D + rrange
        ptr_x2 = x2_ptr + j * D + rrange

        # Load
        val_x1 = tl.load(ptr_x1, mask=rmask & xmask, other=0.0)
        val_x2 = tl.load(ptr_x2, mask=rmask & xmask, other=0.0)

        diff = val_x1 - val_x2
        acc += diff * diff

    # For p_distance=2.0, we apply sqrt
    # Add eps_distance inside the sqrt to avoid zero denominators
    acc_sqrt = tl.sqrt(acc + eps_distance)

    # Store
    tl.store(out_ptr + xindex, acc_sqrt, mask=xmask)

def normalize_pairwise_distance(
    x1, 
    x2, 
    p_distance=2.0, 
    eps_distance=1e-6, 
    keepdim=False, 
    p_norm=2, 
    dim_norm=1, 
    eps_norm=1e-12
):
    """
    Computes the pairwise distance between x1 and x2 using the specified norm (p_distance),
    then normalizes the resulting distances along dim_norm with p_norm.
    """
    # Ensure matching shapes
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")

    if p_distance != 2.0:
        raise NotImplementedError("This implementation currently supports p_distance=2.0 only")

    N, D = x1.shape

    # Prepare output
    out = torch.empty((N, N), device=x1.device, dtype=torch.float32)
    xnumel = N * N
    rnumel = D

    # Heuristic block sizes
    BLOCK_MAX_NB_THREADS = 1024
    RBLOCK = min(triton.next_power_of_2(D), BLOCK_MAX_NB_THREADS)
    XBLOCK = min(BLOCK_MAX_NB_THREADS // RBLOCK, triton.next_power_of_2(N * N))

    grid = ( (xnumel + XBLOCK - 1) // XBLOCK, )

    _pairwise_distance_kernel[grid](
        x1, 
        x2, 
        out,
        N, 
        D, 
        xnumel, 
        rnumel,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        p_distance=p_distance,
        eps_distance=eps_distance
    )

    # Normalize along dim_norm
    # out has shape [N, N], so we apply .norm(p=p_norm, dim=dim_norm)
    norm_vals = out.norm(p=p_norm, dim=dim_norm, keepdim=keepdim)
    out = out / (norm_vals + eps_norm)

    return out
