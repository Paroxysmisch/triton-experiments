import triton
import triton.language as tl

@triton.jit
def relu(x):
    return tl.maximum(x, 0)

@triton.jit
def frac_max_pool2d(input, kernel_size, stride, padding, output_size, return_indices, output_ptr, indices_ptr):
    H, W = input.shape[-2:]
    C = input.shape[-1]
    
    # Compute output dimensions
    out_H = (H + padding * 2 - kernel_size[0]) // stride[0] + 1
    out_W = (W + padding * 2 - kernel_size[1]) // stride[1] + 1
    
    h, w, c = tl.program_id(2), tl.program_id(1), tl.program_id(0)
    
    # Apply ReLU
    input[h, w, c] = relu(input[h, w, c])
    
    # Fractional Max Pooling
    pool_h_start = h * stride[0] - padding[0]
    pool_w_start = w * stride[1] - padding[1]
    pool_h_end = min(pool_h_start + kernel_size[0], H)
    pool_w_end = min(pool_w_start + kernel_size[1], W)
    
    max_val = -float('inf')
    max_idx = -1
    
    for ph in range(pool_h_start, pool_h_end):
        for pw in range(pool_w_start, pool_w_end):
            val = input[ph, pw, c]
            if val > max_val:
                max_val = val
                max_idx = ph * W + pw
    
    output_ptr[c, h, w] = max_val
    if return_indices:
        indices_ptr[c, h, w] = max_idx

@triton.jit
def fused_fractional_max_pool2d_with_relu(input_ptr, kernel_size, output_size, output_ratio, return_indices, output_ptr, indices_ptr):
    N, C, H, W = input_ptr.shape
    
    if output_size is not None:
        out_H, out_W = output_size
    elif output_ratio is not None:
        out_H = int(H * output_ratio[0])
        out_W = int(W * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    
    stride = (out_H // (H + 2 * (kernel_size[0] // 2)), out_W // (W + 2 * (kernel_size[1] // 2)))
    padding = ((kernel_size[0] // 2), (kernel_size[1] // 2))
    
    grid = (C, out_H, out_W)
    block = (1, 1, 1)
    
    frac_max_pool2d[grid](input_ptr, kernel_size, stride, padding, (out_H, out_W), return_indices, output_ptr, indices_ptr)

# Wrapper function
def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if output_size is not None:
        output_size = tuple(output_size)
    elif output_ratio is not None:
        output_ratio = tuple(output_ratio)
    
    N, C, H, W = input.shape
    if output_size is not None:
        out_H, out_W = output_size
    elif output_ratio is not None:
        out_H = int(H * output_ratio[0])
        out_W = int(W * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    
    output = torch.empty((N, C, out_H, out_W), device=input.device, dtype=input.dtype)
    indices = torch.empty((N, C, out_H, out_W), device=input.device, dtype=torch.int32) if return_indices else None
    
    fused_fractional_max_pool2d_with_relu[(N, C, out_H, out_W)](
        input.contiguous().view(N * C, H, W),
        kernel_size,
        output_size,
        output_ratio,
        return_indices,
        output.view(N * C, out_H, out_W),
        indices.view(N * C, out_H, out_W) if return_indices else None
    )
    
    return output if not return_indices else (output, indices)
