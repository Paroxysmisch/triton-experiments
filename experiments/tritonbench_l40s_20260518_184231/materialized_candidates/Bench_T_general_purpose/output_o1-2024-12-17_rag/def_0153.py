import triton
import triton.language as tl
import torch

@triton.jit
def _adaptive_avg_pool2d_kernel(
    inp_ptr, out_ptr,
    B, C, H_in, W_in, H_out, W_out,
    strideN, strideC, strideH, strideW,
    out_strideN, out_strideC, out_strideH, out_strideW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    oh_block = tl.arange(0, BLOCK_M)
    ow_block = tl.arange(0, BLOCK_N)

    oh_start = b_idx * BLOCK_M
    ow_start = c_idx * BLOCK_N

    oh_idx = oh_start + oh_block
    ow_idx = ow_start + ow_block

    # Guard
    oh_mask = oh_idx < H_out
    ow_mask = ow_idx < W_out

    b_offsets = b_idx * strideN
    c_offsets = c_idx * strideC

    # For each valid (oh, ow), compute the corresponding region in input
    # start and end
    oh_fraction_start = oh_idx.to(tl.float32) * (H_in / H_out)
    oh_fraction_end = (oh_idx + 1).to(tl.float32) * (H_in / H_out)
    ow_fraction_start = ow_idx.to(tl.float32) * (W_in / W_out)
    ow_fraction_end = (ow_idx + 1).to(tl.float32) * (W_in / W_out)

    # floor for start indices, ceil (floor) for end
    hs = tl.where(oh_mask, tl.cast(oh_fraction_start, tl.int32), 0)
    he = tl.where(oh_mask, tl.cast(oh_fraction_end, tl.int32), 0)
    ws = tl.where(ow_mask, tl.cast(ow_fraction_start, tl.int32), 0)
    we = tl.where(ow_mask, tl.cast(ow_fraction_end, tl.int32), 0)

    for i_oh in range(BLOCK_M):
        for i_ow in range(BLOCK_N):
            valid = (oh_mask[i_oh] & ow_mask[i_ow])
            if valid:
                h_start = hs[i_oh].item()
                h_end = he[i_oh].item()
                w_start = ws[i_ow].item()
                w_end = we[i_ow].item()

                sum_val = 0.0
                count = 0
                for ih in range(h_start, h_end):
                    for iw in range(w_start, w_end):
                        sum_val += tl.load(
                            inp_ptr + b_offsets + c_offsets
                            + ih * strideH + iw * strideW, mask=True
                        )
                        count += 1
                avg = sum_val / max(count, 1)

                out_offset = b_idx * out_strideN + c_idx * out_strideC
                out_offset += oh_idx[i_oh] * out_strideH + ow_idx[i_ow] * out_strideW
                tl.store(out_ptr + out_offset, avg, mask=True)


def adaptive_avg_pool2d(output_size):
    def _wrapper(input):
        in_dims = input.dim()
        if in_dims == 3:
            # (C, H_in, W_in)
            N = 1
            C, H_in, W_in = input.shape
        else:
            # (N, C, H_in, W_in)
            N, C, H_in, W_in = input.shape

        if isinstance(output_size, int):
            H_out = output_size
            W_out = output_size
        else:
            H_out, W_out = output_size if len(output_size) == 2 else (output_size[0], output_size[0])

        if H_out is None:
            H_out = H_in
        if W_out is None:
            W_out = W_in

        out_shape = (N, C, H_out, W_out) if in_dims == 4 else (C, H_out, W_out)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

        # Prepare launch grid
        grid = (N * H_out // 4 + 1, C * W_out // 4 + 1)
        BLOCK_M = 4
        BLOCK_N = 4

        triton.run(
            _adaptive_avg_pool2d_kernel,
            grid=grid,
            args=[
                input.data_ptr(), out.data_ptr(),
                N, C, H_in, W_in, H_out, W_out,
                input.stride(0) if in_dims == 4 else 0,    # strideN
                input.stride(1) if in_dims == 4 else input.stride(0),  # strideC
                input.stride(-2),                         # strideH
                input.stride(-1),                         # strideW
                out.stride(0) if in_dims == 4 else 0,      # out_strideN
                out.stride(1) if in_dims == 4 else out.stride(0),      # out_strideC
                out.stride(-2),                            # out_strideH
                out.stride(-1),                            # out_strideW
                BLOCK_M, BLOCK_N
            ],
            num_warps=4
        )
        return out
    return _wrapper
