import triton
import triton.language as tl
import torch

@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr,       # float * [N, C_in, H_in, W_in]
    weight_ptr,      # float * [C_out, C_in, KH, KW]
    bias_ptr,        # float * [C_out] or None
    output_ptr,      # float * [N, C_out, H_out, W_out]
    N, C_IN, H_IN, W_IN,
    C_OUT, KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DILATION_H, DILATION_W,
    HAS_BIAS,  # 1 if bias is not None, 0 otherwise
    NEGATIVE_SLOPE,
    NUM_ELEMENTS,    # total number of output elements = N*C_OUT*H_OUT*W_OUT
    BLOCK_SIZE: tl.constexpr
):
    # Compute global indices for the current program.
    # Each index corresponds to one output element: (n, c_out, h_out, w_out)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS

    # Compute the output coordinates for each thread.
    # out_idx: flatten (n, c_out, h_out, w_out)
    # c_out = out_idx // (H_out*W_out*N), but arrangement can differ
    # We'll unravel in this order: n, c_out, h_out, w_out
    # We need to figure out H_out and W_out from the formula:
    #   H_out = (H_in + 2*PAD_H - DILATION_H*(KH-1) - 1)//STRIDE_H + 1
    #   W_out = (W_in + 2*PAD_W - DILATION_W*(KW-1) - 1)//STRIDE_W + 1
    # We'll compute them on-the-fly below.

    # Since we can't pass them separately, let's derive them from NUM_ELEMENTS
    # and the known factors (N, C_OUT).
    # For safe usage, we do a small search (not the best practice, but workable for demonstration).
    # Alternatively, the caller can pass H_OUT, W_OUT as well.
    # We'll approximate them here. 
    # This is a kernel example; in a real scenario, pass H_OUT, W_OUT directly.
    # We'll do a minimal
