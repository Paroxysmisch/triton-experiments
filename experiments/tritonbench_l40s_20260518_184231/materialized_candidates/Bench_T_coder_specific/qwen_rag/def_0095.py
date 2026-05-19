import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr,      # Pointer to input tensor
    running_mean_ptr,  # Pointer to running mean
    running_var_ptr,   # Pointer to running variance
    weight_ptr,      # Pointer to weight tensor (optional)
    bias_ptr,        # Pointer to bias tensor (optional)
    output_ptr,      # Pointer to output tensor
    N, M, K,         # Dimensions of the input tensor (N=batch, M=channels, K=spatial_dims)
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Program ID along the batch dimension
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets[:, None, None] * M * K, mask=mask[:, None, None])
    mu = tl.load(running_mean_ptr + offsets[:, None], mask=mask[:, None])
    var = tl.load(running_var_ptr + offsets[:, None], mask=mask[:, None])

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets[:, None], mask=mask[:, None])
    else:
        weight = 1.0

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets[:, None], mask=mask[:, None])
    else:
        bias = 0.0

    epsilon = 1e-05
    norm_x = (x - mu) / tl.sqrt(var + epsilon)
    output = norm_x * weight + bias

    tl.store(output_ptr + offsets[:, None, None] * M * K, output, mask=mask[:, None, None])
