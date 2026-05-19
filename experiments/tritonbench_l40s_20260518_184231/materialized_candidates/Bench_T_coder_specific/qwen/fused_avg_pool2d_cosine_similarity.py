import triton
from triton.language import *

@triton.jit
def cosine_similarity_kernel(x1_ptr, x2_ptr, output_ptr, batch_size, seq_len, hidden_dim, eps):
    b = tl.program_id(0)
    i = tl.program_id(1)

    if b >= batch_size or i >= seq_len:
        return

    dot_product = 0.0
    norm_x1 = 0.0
    norm_x2 = 0.0

    for j in range(hidden_dim):
        dot_product += x1_ptr[b * seq_len * hidden_dim + i * hidden_dim + j] * \
                       x2_ptr[b * seq_len * hidden_dim + i * hidden_dim + j]
        norm_x1 += x1_ptr[b * seq_len * hidden_dim + i * hidden_dim + j] * \
                    x1_ptr[b * seq_len * hidden_dim + i * hidden_dim + j]
        norm_x2 += x2_ptr[b * seq_len * hidden_dim + i * hidden_dim + j] * \
                    x2_ptr[b * seq_len * hidden_dim + i * hidden_dim + j]

    output_ptr[b * seq_len + i] = dot_product / (tl.sqrt(norm_x1) * tl.sqrt(norm_x2) + eps)

@triton.jit
def avg_pool2d_kernel(input_ptr, output_ptr, batch_size, seq_len, hidden_dim, kernel_size, stride, padding):
    b = tl.program_id(0)
    h_out = tl.program_id(1)
    w_out = tl.program_id(2)

    if b >= batch_size or h_out >= seq_len or w_out >= seq_len:
        return

    sum = 0.0
    count = 0

    for h_in in range(h_out * stride - padding, min(h_out * stride - padding + kernel_size, seq_len)):
        for w_in in range(w_out * stride - padding, min(w_out * stride - padding + kernel_size, seq_len)):
            sum += input_ptr[b * seq_len * hidden_dim + h_in * seq_len + w_in]
            count += 1

    output_ptr[b * seq_len + h_out * seq_len + w_out] = sum / count

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert len(x1.shape) == 3, "Input tensors must be 3-dimensional"

    batch_size, seq_len, hidden_dim = x1.shape
    if stride is None:
        stride = kernel_size

    output_shape = (batch_size, seq_len, seq_len)
    output = torch.empty(output_shape, device=x1.device)

    grid = (batch_size, seq_len, seq_len)
    block = (32, 1, 1)

    cosine_similarity_kernel[grid, block](x1.data_ptr(), x2.data_ptr(), output.data_ptr(), batch_size, seq_len, hidden_dim, eps)
    avg_pool2d_kernel[grid, block](output.data_ptr(), output.data_ptr(), batch_size, seq_len, hidden_dim, kernel_size, stride, padding)

    return output
