import triton
import triton.language as tl
import torch

@triton.jit
def _fused_fract_maxpool2d_relu_kernel(
    input_ptr, output_ptr, indices_ptr,
    N, C, H, W, outH, outW,
    kH, kW,
    strideN, strideC, strideH, strideW,
    strideOutN, strideOutC, strideOutH, strideOutW,
    return_indices: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Each thread corresponds to one output element (n, c, oh, ow)
    # Flatten (n, c, oh, ow) -> idx
    # total = N * C * outH * outW
    mask = pid < (N * C * outH * outW)
    idx = pid
    # Decompose idx back into n, c, oh, ow
    ow = idx % outW
    idx = idx // outW
    oh = idx % outH
    idx = idx // outH
    c = idx % C
    n = idx // C

    # Convert to int32
    ow = ow.to(tl.int32)
    oh = oh.to(tl.int32)
    c = c.to(tl.int32)
    n = n.to(tl.int32)

    # Compute fractional offsets
    # If outH or outW == 1, avoid division by zero
    fracH = (H - kH).to(tl.float32) / (outH - 1).to(tl.float32) if outH > 1 else 0.
    fracW = (W - kW).to(tl.float32) / (outW - 1).to(tl.float32) if outW > 1 else 0.

    row_start = tl.floor(oh.to(tl.float32) * fracH)
    col_start = tl.floor(ow.to(tl.float32) * fracW)

    row_start = tl.max(row_start, 0).to(tl.int32)
    col_start = tl.max(col_start, 0).to(tl.int32)

    max_val = tl.full([BLOCK], -3.4e38, dtype=tl.float32)
    max_idx = tl.zeros([BLOCK], dtype=tl.int32)

    for i in range(256):  # just a compile-time upper bound
        # break if i >= kH
        if i >= kH:
            break
        for j in range(256):  # compile-time upper bound
            if j >= kW:
                break
            r = row_start + i
            c_ = col_start + j
            in_range = (r < H) & (c_ < W) & mask
            idx_in = (n * strideN + c * strideC +
                      r * strideH + c_ * strideW)
            val = tl.where(in_range, tl.load(input_ptr + idx_in, mask=in_range, other=0.), 0.)
            # ReLU
            val = tl.maximum(val, 0.)
            better = val > max_val
            max_val = tl.where(better, val, max_val)
            if return_indices:
                flat_i = (r * W + c_).to(tl.int32)
                max_idx = tl.where(better, flat_i, max_idx)

    # Store results
    out_idx = (n * strideOutN +
               c * strideOutC +
               oh * strideOutH +
               ow * strideOutW)
    tl.store(output_ptr + out_idx, max_val, mask=mask)
    if return_indices:
        tl.store(indices_ptr + out_idx, max_idx, mask=mask)

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False
) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size

    H = input.shape[2]
    W = input.shape[3]

    if output_size is not None:
        outH, outW = output_size
    elif output_ratio is not None:
        outH = int(H * output_ratio[0])
        outW = int(W * output_ratio[1])
    else:
        raise ValueError("Must specify either output_size or output_ratio")

    N, C = input.shape[0], input.shape[1]
    out = torch.empty((N, C, outH, outW), dtype=input.dtype, device=input.device)
    idx_out = None
    if return_indices:
        idx_out = torch.empty((N, C, outH, outW), dtype=torch.int32, device=input.device)

    # Strides
    strideN = input.stride(0)
    strideC = input.stride(1)
    strideH = input.stride(2)
    strideW = input
