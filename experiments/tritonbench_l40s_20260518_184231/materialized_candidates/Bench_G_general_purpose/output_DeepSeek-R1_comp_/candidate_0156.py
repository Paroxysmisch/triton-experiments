import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(
    Out, In, 
    stride_z, stride_m, stride_n,
    N, M, 
    CAUSAL: tl.constexpr, 
    LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    offsets = row_idx * stride_m + col_idx * stride_n + tl.arange(0, DEPTH)
    mask = offsets < N
    
    # Load input with appropriate masking
    if MASK_TYPE == 1:
        causal_mask = (offsets % M) <= (row_idx % M)
        mask &= causal_mask
    x = tl.load(In + offsets, mask=mask, other=-float('inf'))
    
    # Compute softmax
    if IS_FP16:
        x = x.to(tl.float32)
        
    x_minus_max = x - tl.max(x, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    
    if LOG:
        output = x_minus_max - tl.log(denominator)
    else:
        output = numerator / denominator
        
    # Store output
    tl.store(Out + offsets, output.to(Out.dtype.element_ty), mask=mask)

def softmax(input, mask_type=None, causal=False, log=False):
    assert input.dim() == 3
    Z, M, N = input.shape
    
    # Heuristics
    def power_of_2_heuristic(n):
        return triton.next_power_of_2(n)
    
    # Ensure contiguous
    input = input.contiguous()
    output = torch.empty_like(input)
    
    grid = (M * Z, Z)
    _softmax[grid](
        output, input,
        input.stride(0), input.stride(1), input.stride(2),
        N, M,
        CAUSAL=causal,
        LOG=log,
        MASK_TYPE=1 if mask_type is not None else 0,
        DEPTH=power_of_2_heuristic(N),
        IS_FP16=input.dtype == torch.float16,
    )
    return output

@triton.jit
def _softmax_backward(
    GradIn, GradOut, Output,
    stride_z, stride_m, stride_n,
    N, M,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    offsets = row_idx * stride_m + col_idx * stride_n + tl.arange(0, DEPTH)
    mask = offsets < N
    
    # Load data
    grad_out = tl.load(GradOut + offsets, mask=mask, other=0)
    output = tl.load(Output + offsets, mask=mask, other=0)
    
    if IS_FP16:
        grad_out = grad_out.to(tl.float32)
        output = output.to(tl.float32)
    
    # Compute gradient
    if LOG:
        sum_grad = tl.sum(grad_out, axis=0)
        grad = grad_out - tl.exp(output) * sum_grad
    else:
        sum_term = tl.sum(grad_out * output, axis=0)
        grad = output * (grad_out - sum_term)
    
    # Store gradient
    tl.store(GradIn + offsets, grad.to(GradIn.dtype.element_ty), mask=mask)

def softmax_backward(grad_output, output, log=False, causal=False):
    assert grad_output.shape == output.shape
    Z, M, N = output.shape
    
    grad_input = torch.empty_like(grad_output)
    
    grid = (M * Z, Z)
    _softmax_backward[grid](
        grad_input, grad_output, output,
        output.stride(0), output.stride(1), output.stride(2),
        N, M,
        LOG=log,
        CAUSAL=causal,
        DEPTH=triton.next_power_of_2(N),
        IS_FP16=output.dtype == torch.float16,
    )
    return grad_input
