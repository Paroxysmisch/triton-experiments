import torch
import triton
import triton.language as tl

@triton.jit
def _hstack_div_kernel(
    x_ptr,
    d_ptr,
    output_ptr,
    n_elements,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    result = x / d

    if rounding_mode == 1:  # 'trunc'
        result = tl.math.trunc(result)
    elif rounding_mode == 2:  # 'floor'
        result = tl.math.floor(result)

    tl.store(output_ptr + offsets, result, mask=mask)

def stacked_hstack_div_triton(x, d, rounding_mode):
    x = x.contiguous()
    d = d.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rounding_code = 0
    if rounding_mode == 'trunc':
        rounding_code = 1
    elif rounding_mode == 'floor':
        rounding_code = 2
    _hstack_div_kernel[grid](
        x, d, output, n_elements, rounding_code, BLOCK_SIZE=1024
    )
    return output

class FusedHStackDivAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stacked, divisor_broadcasted, rounding_mode):
        ctx.rounding_mode = rounding_mode
        ctx.save_for_backward(stacked, divisor_broadcasted)
        output = stacked_hstack_div_triton(stacked, divisor_broadcasted, rounding_mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rounding_mode = ctx.rounding_mode
        stacked, divisor = ctx.saved_tensors
        grad_stacked = grad_divisor = None

        if ctx.needs_input_grad[0]:
            grad_stacked = grad_output / divisor

        if ctx.needs_input_grad[1]:
            grad_divisor = -grad_output * stacked / (divisor ** 2)
            grad_divisor = grad_divisor.sum(dim=tuple(range(grad_divisor.dim())))

        return grad_stacked, grad_divisor, None, None

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    if len(tensors) == 0:
        raise ValueError("Expected at least one tensor in tensors")
    
    stacked = torch.hstack(tensors)
    divisor_tensor = torch.as_tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    
    try:
        divisor_broadcasted = torch.broadcast_to(divisor_tensor, stacked.shape)
    except RuntimeError as e:
        raise ValueError("Divisor is not broadcastable to the stacked tensor's shape") from e

    stacked = stacked.contiguous()
    divisor_broadcasted = divisor_broadcasted.contiguous()
    
    output = FusedHStackDivAutograd.apply(stacked, divisor_broadcasted, rounding_mode)
    
    if out is not None:
        out.copy_(output)
        return out
    return output
