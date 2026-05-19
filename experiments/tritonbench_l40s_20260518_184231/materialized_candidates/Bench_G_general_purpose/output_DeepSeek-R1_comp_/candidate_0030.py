import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'RBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 64, 'RBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr, gamma_ptr, beta_ptr, mean_ptr, inv_std_ptr, out_ptr,
    xnumel, rnumel, eps,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindices = xoffset + tl.arange(0, XBLOCK)
    xmask = xindices < xnumel

    # Compute mean
    mean = tl.zeros((XBLOCK,), dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindices = roffset + tl.arange(0, RBLOCK)
        rmask = rindices < rnumel
        data = tl.load(
            in_ptr + xindices[:, None] * rnumel + rindices[None, :],
            mask=xmask[:, None] & rmask[None, :],
            other=0.0
        )
        mean += tl.sum(data, axis=1)
    mean = mean / rnumel

    # Compute variance
    var = tl.zeros((XBLOCK,), dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindices = roffset + tl.arange(0, RBLOCK)
        rmask = rindices < rnumel
        data = tl.load(
            in_ptr + xindices[:, None] * rnumel + rindices[None, :],
            mask=xmask[:, None] & rmask[None, :],
            other=0.0
        )
        data_centered = data - mean[:, None]
        var += tl.sum(data_centered * data_centered, axis=1)
    var = var / rnumel
    inv_std = tl.libdevice.rsqrt(var + eps)

    # Store mean and inverse std
    tl.store(mean_ptr + xindices, mean, mask=xmask)
    tl.store(inv_std_ptr + xindices, inv_std, mask=xmask)

    # Normalize and apply gamma + beta
    for roffset in range(0, rnumel, RBLOCK):
        rindices = roffset + tl.arange(0, RBLOCK)
        rmask = rindices < rnumel
        data = tl.load(
            in_ptr + xindices[:, None] * rnumel + rindices[None, :],
            mask=xmask[:, None] & rmask[None, :],
            other=0.0
        )
        data_norm = (data - mean[:, None]) * inv_std[:, None]
        gamma = tl.load(gamma_ptr + rindices, mask=rmask)
        beta = tl.load(beta_ptr + rindices, mask=rmask)
        out_data = data_norm * gamma[None, :] + beta[None, :]
        tl.store(
            out_ptr + xindices[:, None] * rnumel + rindices[None, :],
            out_data,
            mask=xmask[:, None] & rmask[None, :]
        )

def fused_native_layer_norm_no_welford(input, gamma, beta, eps):
    # Ensure contiguous tensors
    input = input.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    M, N = input.shape
    # Allocate output tensors
    output = torch.empty_like(input)
    mean = torch.empty(M, device=input.device, dtype=torch.float32)
    inv_std = torch.empty(M, device=input.device, dtype=torch.float32)

    # Grid definition based on autotuned XBLOCK
    grid = lambda meta: (triton.cdiv(M, meta['XBLOCK']),)
    # Launch kernel
    triton_red_fused_native_layer_norm_no_welford[grid](
        input, gamma, beta, mean, inv_std, output,
        M, N, eps,
    )
    return output, mean, inv_std
