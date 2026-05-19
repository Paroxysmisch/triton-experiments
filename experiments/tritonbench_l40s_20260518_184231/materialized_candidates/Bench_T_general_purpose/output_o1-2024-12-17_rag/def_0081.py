import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    OH, OW,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < (N * C * OH * OW)

    # Calculate n, c, oh, ow from the flat index
    n = idx // (C * OH * OW)
    cc = (idx // (OH * OW)) % C
    oh = (idx // OW) % OH
    ow = idx % OW

    # Compute the corresponding sub-region in the input for adaptive avg pool
    h_start = (oh * H) // OH
    h_end   = ((oh + 1) * H) // OH
    w_start = (ow * W) // OW
    w_end   = ((ow + 1) * W) // OW

    # Load and sum the region
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    area = (h_end - h_start) * (w_end - w_start)

    # Loop over H dimension
    for ih in range(0, 256):  # 256 is a chunk, change as needed
        h_cur = h_start + ih
        if h_cur >= h_end:
            break
        # Loop over W dimension
        for jw in range(0, 256):
            w_cur = w_start + jw
            if w_cur >= w_end:
                break
            # Compute input offset
            in_offset = (
                n * (C * H * W)
                + cc * (H * W)
                + h_cur * W
                + w_cur
            )
            val = tl.load(input_ptr + in_offset, mask=mask)
            sum_val += val

    # Average
    avg_val = sum_val / tl.max(1.0, tl.float32(area))

    # Sigmoid
    out_val = 1.0 / (1.0 + tl.exp(-avg_val))

    # Store
    out_offset = (
        n * (C * OH * OW)
        + cc * (OH * OW)
        + oh * OW
        + ow
    )
    tl.store(output_ptr + out_offset, out_val, mask=mask)


def sigmoid_adaptive_avg_pool2d(input: torch.Tensor,
                                output_size: int or tuple) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    OH, OW = output_size
    N, C, H, W = input.shape
    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    BLOCK = 256
    grid = lambda meta: ((N * C * OH * OW + BLOCK - 1) // BLOCK,)

    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input, out,
        N, C, H, W,
        OH, OW,
        BLOCK=BLOCK
    )
    return out
