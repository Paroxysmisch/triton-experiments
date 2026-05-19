import triton
import triton.language as tl

# Constants defining block sizes for parallelism
XBLOCK = 128  # Number of elements processed in parallel per block in X dimension
RBLOCK = 128  # Number of elements processed in parallel per block in R dimension

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    primals_1_ptr, primals_2_ptr, primals_3_ptr,
    mean_ptr, inv_std_ptr, norm_ptr,
    stride_x, stride_r, N,
    BLOCK: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)

    # Create a block of indices for the reduction dimension
    r_offset = pid * RBLOCK + tl.arange(0, RBLOCK)
    mask_r = r_offset < N

    # Load input data
    primals_1 = tl.load(primals_1_ptr + r_offset * stride_r, mask=mask_r, other=0.0)
    primals_2 = tl.load(primals_2_ptr + r_offset * stride_r, mask=mask_r, other=0.0)
    primals_3 = tl.load(primals_3_ptr + r_offset * stride_r, mask=mask_r, other=0.0)

    # Compute mean
    mean = (primals_1 + primals_2 + primals_3) / 3.0
    tl.store(mean_ptr + r_offset * stride_r, mean, mask=mask_r)

    # Compute variance
    var = ((primals_1 - mean) ** 2 + (primals_2 - mean) ** 2 + (primals_3 - mean) ** 2) / 3.0

    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(inv_std_ptr + r_offset * stride_r, inv_std, mask=mask_r)

    # Normalize inputs
    norm_1 = (primals_1 - mean) * inv_std
    norm_2 = (primals_2 - mean) * inv_std
    norm_3 = (primals_3 - mean) * inv_std

    # Store normalized outputs
    tl.store(norm_ptr + r_offset * stride_r, norm_1, mask=mask_r)
    tl.store(norm_ptr + r_offset * stride_r + stride_x, norm_2, mask=mask_r)
    tl.store(norm_ptr + r_offset * stride_r + 2 * stride_x, norm_3, mask=mask_r)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    # Assume primals_1, primals_2, and primals_3 are Triton-compatible tensors
    N = primals_1.shape[0]
    stride_x = primals_1.stride(0)
    stride_r = primals_1.stride(1)

    # Allocate output tensors
    mean = torch.empty_like(primals_1)
    inv_std = torch.empty_like(primals_1)
    norm = torch.empty((3, *primals_1.shape), dtype=primals_1.dtype, device=primals_1.device)

    # Launch Triton kernel
    grid = (triton.cdiv(N, RBLOCK),)
    triton_red_fused_native_layer_norm_no_welford[grid](
        primals_1, primals_2, primals_3,
        mean, inv_std, norm,
        stride_x, stride_r, N,
        BLOCK=RBLOCK
    )

    return norm, mean, inv_std
