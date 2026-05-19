import triton
import triton.language as tl

BLOCK_SIZE = 128
NUM_WARPS = 4

@triton.jit
def _rms_layernorm_forward(
    x_ptr, weight_ptr, output_ptr, variance_ptr, num_rows, num_cols, block_size: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * block_size
    row_end = min(row_start + block_size, num_rows)
    
    # Compute row-wise variance
    var = 0.0
    for col in range(num_cols):
        var += tl.dot(x_ptr[row_start + col * num_rows], x_ptr[row_start + col * num_rows])
    var = tl.sum(var, axis=0) / num_cols
    variance_ptr[row] = var
    
    # Compute inverse square root of variance
    inv_sqrt_var = 1.0 / tl.sqrt(variance_ptr[row] + 1e-6)
    
    # Normalize and scale the input
    for col in range(num_cols):
        output_ptr[row_start + col * num_rows] = weight_ptr[col] * x_ptr[row_start + col * num_rows] * inv_sqrt_var

@triton.jit
def _rms_layernorm_backward(
    x_ptr, weight_ptr, output_ptr, variance_ptr, grad_output_ptr, grad_x_ptr, grad_weight_ptr, num_rows, num_cols, block_size: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * block_size
    row_end = min(row_start + block_size, num_rows)
    
    # Compute gradients with respect to input
    inv_sqrt_var = 1.0 / tl.sqrt(variance_ptr[row] + 1e-6)
    for col in range(num_cols):
        grad_x_ptr[row_start + col * num_rows] = weight_ptr[col] * grad_output_ptr[row_start + col * num_rows] * inv_sqrt_var
    
    # Compute gradients with respect to weight
    for col in range(num_cols):
        grad_weight_ptr[col] = tl.sum(x_ptr[row_start + col * num_rows] * grad_output_ptr[row_start + col * num_rows], axis=0) * inv_sqrt_var

@triton.jit
def _gemma_rms_layernorm_forward(
    x_ptr, weight_ptr, output_ptr, variance_ptr, num_rows, num_cols, block_size: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * block_size
    row_end = min(row_start + block_size, num_rows)
    
    # Compute row-wise variance
    var = 0.0
    for col in range(num_cols):
        var += tl.dot(x_ptr[row_start + col * num_rows], x_ptr[row_start + col * num_rows])
    var = tl.sum(var, axis=0) / num_cols
    variance_ptr[row] = var
    
    # Compute inverse square root of variance
    inv_sqrt_var = 1.0 / tl.sqrt(variance_ptr[row] + 1e-6)
    
    # Normalize and scale the input with an additional constant
    for col in range(num_cols):
        output_ptr[row_start + col * num_rows] = (weight_ptr[col] + 1.0) * x_ptr[row_start + col * num_rows] * inv_sqrt_var

@triton.jit
def calculate_settings(num_rows, num_cols):
    return BLOCK_SIZE, NUM_WARPS

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        num_rows, num_cols = x.shape
        block_size, num_warps = calculate_settings(num_rows, num_cols)
        
        variance = torch.zeros(num_rows, device=x.device, dtype=x.dtype)
        output = torch.zeros_like(x)
        
        _rms_layernorm_forward[
            grid=(num_rows // block_size, num_cols // block_size),
            block=(block_size, block_size, 1),
            num_warps=num_warps,
        ](x, weight, output, variance, num_rows, num_cols, block_size)
        
        ctx.save_for_backward(x, weight, variance)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, variance = ctx.saved_tensors
        num_rows, num_cols = x.shape
        block_size, num_warps = calculate_settings(num_rows, num_cols)
        
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)
        grad_variance = torch.zeros_like(variance)
        
        _rms_layernorm_backward[
            grid=(num_rows // block_size, num_cols // block_size),
            block=(block_size, block_size, 1),
            num_warps=num_warps,
        ](x, weight, grad_output, variance, grad_output, grad_x, grad_weight, num_rows, num_cols, block_size)
        
        return grad_x, grad_weight

def fast_rms_layernorm(x, weight):
    return Fast_RMS_Layernorm.apply(x, weight)
