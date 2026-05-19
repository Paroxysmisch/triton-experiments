import triton
import triton.language as tl

@triton.jit
def _sigmoid_conv2d_kernel(
    input_ptr,         # [N, C_in, H_in, W_in]
    weight_ptr,        # [C_out, C_in/groups, kH, kW]
    bias_ptr,          # [C_out] or None
    output_ptr,        # [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in,
    C_out, kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    # Launch grid/meta
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one output element or a tile of output elements.
    # For simplicity, handle row-major indexing for (N * C_out * H_out * W_out).

    # Flattened output volume
    out_total_elems = N * C_out * ((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h + 1) \
                              * ((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w + 1)
    # Index in flattened output
    idx = pid * BLOCK_M
    # We'll handle a small chunk of output
    for i_off in range(BLOCK_M):
        o_idx = idx + i_off
        if o_idx >= out_total_elems:
            return
        # Decompose out index into N, C_out, H_out, W_out
        w_out = o_idx % (((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w) + 1)
        h_out = (o_idx // (((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w + 1))) \
                % (((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h) + 1)
        co    = (o_idx // ((((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w + 1) *
                           (((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h) + 1))) \
                % C_out
        n_idx = o_idx // (C_out * (((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w + 1) *
                                   (((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h) + 1)))

        # Compute convolution
        # group of out_channel means co belongs to which group
        group_idx = co // (C_out // groups)
        c_per_group = C_in // groups

        tmp = tl.float32(0.)
        # Indices in input
        out_y = h_out * stride_h - pad_h
        out_x = w_out * stride_w - pad_w
        # Weight pointer offset for the out channel
        weight_offset = co * c_per_group * kH * kW
        # Accumulate
        for ci in range(c_per_group):
            for ky in range(kH):
                in_y = out_y + ky * dil_h
                if (in_y < 0) or (in_y >= H_in):
                    continue
                for kx in range(kW):
                    in_x = out_x + kx * dil_w
                    if (in_x < 0) or (in_x >= W_in):
                        continue
                    # Input channel index
                    in_c = group_idx * c_per_group + ci

                    # input index in flattened form
                    in_index = (n_idx * C_in * H_in * W_in) + (in_c * H_in * W_in) + (in_y * W_in) + in_x
                    # weight index
                    w_index = weight_offset + (ci * kH * kW) + (ky * kW) + kx

                    val_in = tl.load(input_ptr + in_index)
                    val_wt = tl.load(weight_ptr + w_index)
                    tmp += val_in * val_wt

        # Add bias if provided
        if bias_ptr != 0:
            tmp += tl.load(bias_ptr + co)

        # Sigmoid activation
        tmp = 1.0 / (1.0 + tl.exp(-tmp))

        # Store result
        out_index = (n_idx * C_out * (((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h) + 1) *
                                   (((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w) + 1)) \
                    + (co * (((H_in + 2 * pad_h - dil_h*(kH-1) - 1)//stride_h) + 1) *
                          (((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w) + 1)) \
                    + (h_out * ((((W_in + 2 * pad_w - dil_w*(kW-1) - 1)//stride_w) + 1))) \
                    + w_out
        tl.store(output_ptr + out_index, tmp)


def sigmoid_conv2d(input, 
                   weight, 
                   bias=None, 
                   stride=1, 
                   padding=0, 
                   dilation=1, 
                   groups=1, 
                   out=None):
    """
    sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None) -> Tensor
    
    Applies a 2D convolution over an input tensor with the given filters, then applies the sigmoid
    activation element-wise. 

    Args:
        input (Tensor): The input tensor of shape (N, C_in, H_in, W_in).
        weight (Tensor): The convolution filters of shape (C_out, C_in/groups, kH, kW).
        bias (Tensor, optional): Optional bias of shape (C_out). Default: None.
        stride (int or tuple, optional): The stride of the convolving kernel. Default: 1.
        padding (int, tuple, or string, optional): Implicit padding on both sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        out (Tensor, optional): Output tensor.

    Returns:
        Tensor: Result of convolution followed by a sigmoid activation, shape (N, C_out, H_out, W_out).
    """

    # Convert stride, padding, dilation to (h, w) for internal usage
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif isinstance(padding, str):
        # Basic 'same' or 'valid' handling demonstration
        # This simplistic approach assumes stride=1 for "same"
        if padding.lower() == 'same':
            # For a "same" conv with stride=1:
            # out_dim = ceil(in_dim / stride)
            # pad = (out_dim - 1) // 2
            pad_h = (input.shape[2] - 1) // 2
            pad_w = (input.shape[3] - 1) // 2
        else:  # 'valid'
            pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = 0, 0

    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = dilation

    N, C_in, H_in, W_in = input.shape
    C_out, _, kH, kW = weight.shape

    # Compute output dims
    H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    if out is None:
        import torch
        out = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)

    # Bias pointer
    bias_ptr = 0
    if bias is not None:
        bias_ptr = bias.data_ptr()

    # We'll launch one program per output element in naive fashion.
    # The kernel uses a single program_id(0). We just set the grid appropriately.
    out_elems = N * C_out * H_out * W_out
    BLOCK_M = 1  # each program handles one output element (simplistic)
    grid = lambda meta: ( (out_elems + meta['BLOCK_M'] - 1) // meta['BLOCK_M'], )

    _sigmoid_conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias_ptr if bias is not None else 0,
        output_ptr=out.data_ptr(),
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, kH=kH, kW=kW,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dil_h=dil_h, dil_w=dil_w,
        groups=groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=1
    )

    return out
