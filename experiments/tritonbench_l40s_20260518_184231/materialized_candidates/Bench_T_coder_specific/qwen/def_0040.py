import triton
import triton.language as tl

@triton.jit
def batch_norm_sigmoid_kernel(
    X_ptr,
    running_mean_ptr,
    running_var_ptr,
    gamma_ptr,
    beta_ptr,
    Y_ptr,
    N,
    C,
    L,
    training,
    momentum,
    eps,
    block_size: tl.constexpr
):
    # Define indices
    n = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)

    # Load data
    x = tl.load(X_ptr + ((n * C + c) * L + l) * sizeof(T))
    running_mean = tl.load(running_mean_ptr + c * sizeof(T))
    running_var = tl.load(running_var_ptr + c * sizeof(T))
    gamma = tl.load(gamma_ptr + c * sizeof(T)) if gamma_ptr else T(1)
    beta = tl.load(beta_ptr + c * sizeof(T)) if beta_ptr else T(0)

    # Compute mean and variance
    if training:
        local_sum = tl.sum(x, axis=0)
        local_count = tl.full_like(local_sum, N * L, dtype=T)
        new_running_mean = running_mean * (1 - momentum) + local_sum * momentum / local_count
        new_running_var = running_var * (1 - momentum) + (tl.sum((x - local_sum / local_count) ** 2, axis=0) / local_count) * momentum
        tl.store(running_mean_ptr + c * sizeof(T), new_running_mean)
        tl.store(running_var_ptr + c * sizeof(T), new_running_var)
        mean = local_sum / local_count
        var = (tl.sum((x - mean) ** 2, axis=0) / local_count) + eps
    else:
        mean = running_mean
        var = running_var + eps

    # Normalize and apply sigmoid
    norm_x = (x - mean) / tl.sqrt(var)
    y = gamma * norm_x + beta
    sigmoid_y = 1 / (1 + tl.exp(-y))

    # Store result
    tl.store(Y_ptr + ((n * C + c) * L + l) * sizeof(T), sigmoid_y)

# Wrapper function
def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> Tensor:
    assert len(input.shape) == 2 or len(input.shape) == 3, "Input must be of shape (N, C) or (N, C, L)"
    N, C = input.shape[:2]
    L = input.shape[2] if len(input.shape) == 3 else 1
    
    # Create output tensor
    output = torch.zeros_like(input)
    
    # Configure grid and block sizes
    grid = (N, C, L)
    block = (block_size, 1, 1)
    
    # Launch kernel
    batch_norm_sigmoid_kernel[grid, block](
        input.data_ptr(),
        running_mean.data_ptr(),
        running_var.data_ptr(),
        weight.data_ptr() if weight else None,
        bias.data_ptr() if bias else None,
        output.data_ptr(),
        N,
        C,
        L,
        training,
        momentum,
        eps,
        block_size
    )
    
    return output
