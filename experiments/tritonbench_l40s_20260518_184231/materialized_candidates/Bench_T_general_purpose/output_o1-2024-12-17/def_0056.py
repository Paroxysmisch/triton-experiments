import torch
import triton
import triton.language as tl

def _compute_intervals(input_size: int, output_size: int):
    step = (input_size - output_size) / float(output_size)
    intervals = []
    start = 0.0
    for i in range(output_size):
        end = start + kernel_size_d
        interval_start = int(round(start))
        interval_end = int(round(end)) if int(round(end)) > interval_start else interval_start + 1
        intervals.append((interval_start, interval_end))
        start += step + kernel_size_d
    return intervals

@triton.jit
def _fractional_max_pool2d_relu_kernel(
    input_ptr, output_ptr, indices_ptr,
    intervals_h_ptr, intervals_w_ptr,
    stride_h, stride_w,
    in_h, in_w,
    out_h, out_w,
    channels, return_indices: tl.constexpr
):
    c = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    if c >= channels or oh >= out_h or ow >= out_w:
        return

    # Load intervals
    h_start = tl.load(intervals_h_ptr + oh * 2)
    h_end = tl.load(intervals_h_ptr + oh * 2 + 1)
    w_start = tl.load(intervals_w_ptr + ow * 2)
    w_end = tl.load(intervals_w_ptr + ow * 2 + 1)

    max_val = tl.float32(-1e20)
    max_idx = tl.int32(0)
    for ih in range(h_start, h_end):
        for iw in range(w_start, w_end):
            idx = c * in_h * in_w + ih * in_w + iw
            val = tl.load(input_ptr + idx)
            val = tl.where(val > 0.0, val, 0.0)
            if val > max_val:
                max_val = val
                max_idx = idx

    out_idx = c * out_h * out_w + oh * out_w + ow
    tl.store(output_ptr + out_idx, max_val)
    if return_indices:
        tl.store(indices_ptr + out_idx, max_idx)

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False
) -> torch.Tensor:
    if isinstance(kernel_size, int):
        ks_h = ks_w = kernel_size
    else:
        ks_h, ks_w = kernel_size

    in_b, in_c, in_h, in_w = input.shape
    if output_size is None and output_ratio is None:
        raise ValueError("Either output_size or output_ratio must be provided.")
    if output_size is not None:
        out_h, out_w = output_size
    else:
        out_h = int(input.shape[2] * output_ratio[0])
        out_w = int(input.shape[3] * output_ratio[1])

    global kernel_size_d
    kernel_size_d = ks_h  # Reuse for interval calculation on height (simplistic approach)
    intervals_h = _compute_intervals(in_h, out_h)
    kernel_size_d = ks_w  # Reuse for interval calculation on width (simplistic approach)
    intervals_w = _compute_intervals(in_w, out_w)

    intervals_h_tensor = torch.tensor(intervals_h, dtype=torch.int32, device=input.device).flatten()
    intervals_w_tensor = torch.tensor(intervals_w, dtype=torch.int32, device=input.device).flatten()

    out = torch.zeros((in_b, in_c, out_h, out_w), device=input.device, dtype=input.dtype)
    idx_out = None
    if return_indices:
        idx_out = torch.zeros_like(out, dtype=torch.int32, device=input.device)

    grid = (in_c, out_h, out_w)
    for b in range(in_b):
        _fractional_max_pool2d_relu_kernel[grid](
            input[b].contiguous(),
            out[b],
            idx_out[b] if return_indices else 0,
            intervals_h_tensor, intervals_w_tensor,
            ks_h, ks_w,
            in_h, in_w,
            out_h, out_w,
            in_c,
            return_indices=return_indices
        )

    if return_indices:
        return out, idx_out
    return out
