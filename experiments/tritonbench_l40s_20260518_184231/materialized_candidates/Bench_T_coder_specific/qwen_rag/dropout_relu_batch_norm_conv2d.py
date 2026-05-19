import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    N, C_in, H_in, W_in, C_out, K_h, K_w, S, P, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = (N * C_out * (H_in + 2*P - K_h + 1) // S) // BLOCK_SIZE
    x = pid % (N * C_out * (H_in + 2*P - K_h + 1) // S)
    n = x // (C_out * (H_in + 2*P - K_h + 1) // S)
    c_out = (x % (C_out * (H_in + 2*P - K_h + 1) // S)) // ((H_in + 2*P - K_h + 1) // S)
    h_out = (x % ((H_in + 2*P - K_h + 1) // S)) % ((H_in + 2*P - K_h + 1) // S)
    w_out = h_out % ((W_in + 2*P - K_w + 1) // S)

    h_in = h_out * S - P + D
    w_in = w_out * S - P + D

    acc = 0.0
    for kh in range(K_h):
        for kw in range(K_w):
            h_in_ = h_in + kh
            w_in_ = w_in + kw
            if h_in_ >= 0 and h_in_ < H_in and w_in_ >= 0 and w_in_ < W_in:
                i = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_ * W_in + w_in_
                j = c_out * K_h * K_w + kh * K_w + kw
                acc += tl.load(input_ptr + i) * tl.load(weight_ptr + j)
    
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)

    o = n * C_out * ((H_in + 2*P - K_h + 1) // S) + c_out * ((H_in + 2*P - K_h + 1) // S) + h_out * ((W_in + 2*P - K_w + 1) // S) + w_out
    tl.store(output_ptr + o, acc)
