import torch
import triton
import triton.language as tl

# Define the Triton kernel signature
@triton.jit
def combined_activation_kernel(
    input_ptr: tl.tensor,
    weight1_ptr: tl.tensor,
    weight2_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,
    batch_size: tl.int32,
    N: tl.int32,
    D_in: tl.int32,
    D_out: tl.int32,
    BLOCK_SIZE: tl.constexpr(128),
):
    pid = tl.program_id(axis=0)
    coords = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    coords = coords % (batch_size * N * D_out)

    batch_idx = coords // (N * D_out)
    n_idx = (coords // D_out) % N
    d_out_idx = coords % D_out

    xw1 = tl.zeros((D_in,), dtype=tl.float32)
    for j in range(D_in):
        xw1[j] = 0.0
        for k in range(D_in):
            xw1[j] += input_ptr[batch_idx * N * D_in + n_idx * D_in + k] * weight1_ptr[k * D_out + d_out_idx]

    sigmoid_xw1 = 1.0 / (1.0 + tl.exp(-xw1))
    tanh_sigmoid_xw1 = (tl.exp(sigmoid_xw1) - tl.exp(-sigmoid_xw1)) / (tl.exp(sigmoid_xw1) + tl.exp(-sigmoid_xw1))

    ewm = tl.zeros((D_in,), dtype=tl.float32)
    for j in range(D_in):
        ewm[j] = tanh_sigmoid_xw1[j] * weight2_ptr[d_out_idx * D_in + j]

    output_ptr[batch_idx * N * D_out + n_idx * D_out + d_out_idx] = 0.0
    for j in range(D_in):
        output_ptr[batch_idx * N * D_out + n_idx * D_out + d_out_idx] += ewm[j] + bias_ptr[d_out_idx]

# Define the wrapper function
def combined_activation(input, weight1, weight2, bias, out=None):
    if out is None:
        out = torch.empty_like(input)

    batch_size, N, D_in = input.shape
    _, _, D_out = weight1.shape

    # Launch Triton kernel
    grid = (batch_size * N * D_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    combined_activation_kernel[grid, BLOCK_SIZE](
        input.contiguous().data_ptr(),
        weight1.contiguous().data_ptr(),
        weight2.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        batch_size,
        N,
        D_in,
        D_out,
        BLOCK_SIZE,
    )

    return out

# Example usage
if __name__ == "__main__":
    input = torch.randn(2, 3, 4, device='cuda')
    weight1 = torch.randn(4, 5, device='cuda')
    weight2 = torch.randn(5, 3, device='cuda')
    bias = torch.randn(3, device='cuda')

    result = combined_activation(input, weight1, weight2, bias)
    print(result)
