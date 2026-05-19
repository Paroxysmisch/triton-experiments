import torch
import triton
import triton.language as tl

@triton.jit
def _affine_grid_sample_kernel(
    input_ptr,       # float*  (N*C*H_in*W_in total elements)
    theta_ptr,       # float*  (N*2*3 total elements)
    output_ptr,      # float*  (N*C*H_out*W_out total elements)
    N, C, H_in, W_in, H_out, W_out,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,            # 0=bilinear, 1=nearest, 2=bicubic (placeholder)
    PADDING_MODE: tl.constexpr,    # 0=zeros, 1=border, 2=reflection (placeholder)
    ALIGN_CORNERS: tl.constexpr    # bool
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total_out = N * C * H_out * W_out
    mask = offsets < total_out

    # Decompose the linear index into n, c, h, w
    n_chw = tl.where(mask, offsets, 0)
    n = n_chw // (C * H_out * W_out)
    rem = n_chw % (C * H_out * W_out)
    c = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    out_h = rem // W_out
    out_w = rem % W_out

    # Convert output pixel (out_h, out_w) to normalized coords in [-1,1]
    if ALIGN_CORNERS:
        nh = (2 * out_h.float()) / (H_out - 1) - 1
        nw = (2 * out_w.float()) / (W_out - 1) - 1
    else:
        nh = (2 * out_h.float() + 1) / (H_out * 1.0) - 1
        nw = (2 * out_w.float() + 1) / (W_out * 1.0) - 1

    # Load affine theta for this sample n
    # theta shape is (N, 2, 3), so offset is n*6
    t_base = n * 6
    t0 = tl.load(theta_ptr + t_base + 0)  # a11
    t1 = tl.load(theta_ptr + t_base + 1)  # a12
    t2 = tl.load(theta_ptr + t_base + 2)  # a13
    t3 = tl.load(theta_ptr + t_base + 3)  # a21
    t4 = tl.load(theta_ptr + t_base + 4)  # a22
    t5 = tl.load(theta_ptr + t_base + 5)  # a23

    # Apply 2D affine transform
    in_w = t0 * nw + t1 * nh + t2
    in_h = t3 * nw + t4 * nh + t5

    # Map from normalized coords in [-1,1] to input coordinate space
    if ALIGN_CORNERS:
        # x' = (in_w+1)*(W_in-1)/2
        # y' = (in_h+1)*(H_in-1)/2
        real_w = 0.5 * (in_w + 1.0) * (W_in - 1)
        real_h = 0.5 * (in_h + 1.0) * (H_in - 1)
    else:
        # x' = ((in_w+1)*W_in-1)/2
        # y' = ((in_h+1)*H_in-1)/2
        real_w = 0.5 * ((in_w + 1.0) * W_in - 1.0)
        real_h = 0.5 * ((in_h + 1.0) * H_in - 1.0)

    # Depending on interpolation mode
    if MODE == 0:  # bilinear
        # floor and ceil
        w0 = tl.floor(real_w)
        w1 = w0 + 1
        h0 = tl.floor(real_h)
        h1 = h0 + 1

        # get fractional part
        fw = real_w - w0
        fh = real_h - h0

        # clamp coords or handle padding mode
        def clamp_or_pad(x, limit):
            if PADDING_MODE == 0:  # zeros
                return tl.where((x >= 0) & (x < limit), x, -1)
            elif PADDING_MODE == 1:  # border
                return tl.max(tl.zeros_like(x), tl.min(x, limit - 1))
            else:  # reflection placeholder
                # reflect naive
                x_mod = (x + limit * 10000) % (2 * limit)
                x_reflect = tl.where(x_mod < limit, x_mod, 2 * limit - x_mod - 1)
                return x_reflect

        w0c = clamp_or_pad(w0, W_in)
        w1c = clamp_or_pad(w1, W_in)
        h0c = clamp_or_pad(h0, H_in)
        h1c =
