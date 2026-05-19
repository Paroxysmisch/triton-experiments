import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    bias_shape,
    stride,
    padding,
    dilation,
    groups,
    n,
    c_in,
    h_in,
    w_in,
    c_out,
    k_h,
    k_w,
    o_h,
    o_w,
    dtype):
    
    pid = tl.program_id(axis=0)
    row = pid // o_w
    col = pid % o_w
    
    # Compute indices for the input tensor
    r_start = row * stride - padding
    r_end = r_start + k_h
    c_start = col * stride - padding
    c_end = c_start + k_w
    
    acc = tl.zeros((c_out,), dtype=dtype)
    
    for kh in range(k_h):
        kr = r_start + kh
        if kr < 0 or kr >= h_in:
            continue
        for kw in range(k_w):
            kc = c_start + kw
            if kc < 0 or kc >= w_in:
                continue
            for ci in range(c_in):
                for oc in range(0, c_out, groups):
                    group_idx = ci // (c_in // groups)
                    acc[oc] += input_ptr[(n, ci, kr, kc)] * weight_ptr[(oc, ci // (c_in // groups), kh, kw)]
    
    if bias_ptr is not None:
        for oc in range(c_out):
            acc[oc] += bias_ptr[oc]
    
    output_ptr[row * o_w + col] = acc[pid % c_out]
