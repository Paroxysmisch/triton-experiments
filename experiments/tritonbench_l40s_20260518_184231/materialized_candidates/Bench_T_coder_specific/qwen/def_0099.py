triton
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = X[offsets]
    
    # Apply GELU activation
    if approximate == "none":
        cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        y = x * cdf
    else:
        y = 0.5 * x * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (x + 0.044715 * x ** 3)))
    
    # Write back the result
    Y[offsets] = y


@triton.jit
def std_kernel(
    X,
    Y,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = X[offsets]
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / M
    
    # Calculate variance
    variance = tl.sum((x - mean) ** 2, axis=0) / M
    
    # Calculate standard deviation
    std_dev = tl.sqrt(variance) * correction
    
    # Write back the result
    Y[offsets] = std_dev


@triton.jit
def gelu_std_kernel(
    X,
    Y,
    Z,
    N,
    M,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    block_start_x = pid_x * BLOCK_SIZE_X
    block_start_y = pid_y * BLOCK_SIZE_Y
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Apply GELU activation
    if approximate == "none":
        cdf = 0.5 * (1.0 + tl.erf(X[offsets_x] / tl.sqrt(2.0)))
        Y[offsets_x] = X[offsets_x] * cdf
    else:
        Y[offsets_x] = 0.5 * X[offsets_x] * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (X[offsets_x] + 0.044715 * X[offsets_x] ** 3)))
    
    # Calculate mean
    mean = tl.sum(Y[offsets_x], axis=0) / M
    
    # Calculate variance
    variance = tl.sum((Y[offsets_x] - mean) ** 2, axis=0) / M
    
    # Calculate standard deviation
    std_dev = tl.sqrt(variance) * correction
    
    # Write back the result
    Z[offsets_y] = std_dev


# Wrapper function
def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    input_size = input.shape
    num_elements = np.prod(input_size)
    block_size = 1024
    
    if dim is None:
        dim = range(len(input_size))
    
    if isinstance(dim, int):
        dim = [dim]
    
    num_dims = len(dim)
    
    if num_dims == 1:
        output_size = list(input_size)
        output_size[dim[0]] = 1 if keepdim else 0
        output_size = tuple(output_size)
        
        Y = torch.empty(output_size, dtype=input.dtype, device=input.device)
        
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](input, Y, num_elements, BLOCK_SIZE=block_size)
        
        return Y
    elif num_dims > 1:
        raise ValueError("Multi-dimensional reduction not supported yet.")
