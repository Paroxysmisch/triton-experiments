import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for 2D Convolution (naive example).
# This kernel assumes:
#   1) Each program handles one output channel at a time within a tile.
#   2) Weight and bias broadcast are handled in a naive way.
# This is for illustrative purposes. More sophisticated tiling/parallelization can be used for performance.
# ------------------------------------------------------------------------------
@triton.jit
def _conv2d_kernel(
    in_ptr,        # *float32
    wt_ptr,        # *float32
    bias_ptr,      # *float32
    out_ptr,       # *float32
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, K_H, K_W,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    out_h, out_w,
    BLOCK_M: tl.constexpr,  # tile size in M dimension
    BLOCK_N: tl.constexpr,  # tile size in N dimension
):
    # Program ID for output channels and spatial location
    oh = tl.program_id(0)
    oc = tl.program_id(1)
    b  = tl.program_id(2)

    # Create ranges for partial accumulation
    # Each program processes BLOCK_M x BLOCK_N in output's heightxwidth
    oh_range = oh * BLOCK_M + tl.arange(0, BLOCK_M)
    ow_range = tl.arange(0, BLOCK_N)

    # Check valid region within the output image
    valid_oh = oh_range < out_h
    # The kernel processes along width dimension in a loop for demonstration
    for ow_block in range(0, out_w, BLOCK_N):
        ow_val = ow_block + ow_range
        valid_ow = ow_val < out_w

        # Initialize accumulator
        accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Perform the convolution
        for ic in range(IN_C):
            for kh in range(K_H):
                for kw in range(K_W):
                    in_h = (oh_range * stride_h) - pad_h + kh * dil_h
                    in_w = (ow_val * stride_w) - pad_w + kw * dil_w

                    # Load input if in valid range
                    in_valid = (in_h >= 0) & (in_h < IN_H) & (in_w >= 0) & (in_w < IN_W)
                    in_idx = b * (IN_C * IN_H * IN_W) + ic * (IN_H * IN_W) + in_h * IN_W + in_w
                    inp = tl.where(in_valid & valid_oh[:, None] & valid_ow[None, :],
                                   tl.load(in_ptr + in_idx, mask=in_valid[:, None] & valid_oh[:, None] & valid_ow[None, :], other=0.0),
                                   0.0)

                    wt_idx = oc * (IN_C * K_H * K_W) + ic * (K_H * K_W) + kh * K_W + kw
                    w_val = tl.load(wt_ptr + wt_idx)
                    accum += inp * w_val

        # Add bias if provided
        if tl.static_not_none(bias_ptr):
            b_val = tl.load(bias_ptr + oc)
            accum += b_val

        # Store results
        out_idx_base = b * (OUT_C * out_h * out_w) + oc * (out_h * out_w)
        for i in range(BLOCK_M):
            oh_i = oh_range[i]
            if valid_oh[i]:
                for j in range(BLOCK_N):
                    ow_j = ow_val[j]
                    if valid_ow[j]:
                        out_idx = out_idx_base + oh_i * out_w + ow_j
                        tl.store(out_ptr + out_idx, accum[i, j])

# ------------------------------------------------------------------------------
# Pixel Shuffle + Conv2D Wrapper in Python
# ------------------------------------------------------------------------------
def pixel_shuffle_conv2d(input: torch.Tensor,
                         weight: torch.Tensor,
                         bias=None,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         upscale_factor=2) -> torch.Tensor:
    """
    Applies a 2D convolution followed by pixel shuffle upscaling.
    """
    # Extract shapes
    N, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weight.shape

    # Calculate output spatial dimensions for convolution
    out_h = (H_in + 2*padding - dilation*(K_h-1) - 1)//stride + 1
    out_w = (W_in + 2*padding - dilation*(K_w-1) - 1)//stride + 1

    # Allocate output tensor for convolution
    conv_out = torch.empty((N, C_out, out_h, out_w), device=input.device, dtype=input.dtype)

    # Launch kernel
    # We'll launch with a 2D grid for (output height // blockM, output channels, batch)
    # and we tile over the output width in the kernel itself.
    BLOCK_M = 8
    BLOCK_N = 8
    grid = ( (out_h + BLOCK_M - 1)//BLOCK_M, C_out, N )

    # Prepare bias pointer (or None)
    bias_ptr = bias.data_ptr() if bias is not None else None

    _conv2d_kernel[grid](
        input.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        conv_out.data_ptr(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride if isinstance(stride, int) else stride[0],  # only handle int or (int,int)
        stride if isinstance(stride, int) else stride[1],
        padding if isinstance(padding, int) else padding[0],
        padding if isinstance(padding, int) else padding[1],
        dilation if isinstance(dilation, int) else dilation[0],
        dilation if isinstance(dilation, int) else dilation[1],
        out_h, out_w,
        BLOCK_M, BLOCK_N,
        num_warps=1,
        num_stages=1
    )

    # Pixel shuffle
    # C_out should be divisible by (upscale_factor**2)
    # out shape: (N, c_out // r^2, out_h * r, out_w * r)
    r = upscale_factor
    assert C_out % (r*r) == 0, "Output channels must be divisible by upscale_factor^2"
    conv_out_reshaped = conv_out.reshape(N, C_out // (r*r), r, r, out_h, out_w)
    # rearrange
    out = conv_out_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(
        N, C_out // (r*r), out_h * r, out_w * r
    )

    return out
