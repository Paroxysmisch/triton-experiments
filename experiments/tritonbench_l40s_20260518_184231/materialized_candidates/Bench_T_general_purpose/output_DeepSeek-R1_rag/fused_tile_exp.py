import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

class FusedTileExpAutograd(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, dims):
        # Adjust dims to match input dimensions by prepending ones if necessary
        adjusted_dims = (1,) * (input.ndim - len(dims)) + tuple(dims)
        output_shape = tuple(inp_dim * dim for inp_dim, dim in zip(input.shape, adjusted_dims))
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        
        # Flatten the input and output to 1D for processing
        input_flat = input.contiguous().view(-1)
        output_flat = output.view(-1)
        input_numel = input_flat.numel()
        output_numel = output_flat.numel()
        
        # Launch the kernel
        grid = lambda meta: (triton.cdiv(output_numel, meta['BLOCK_SIZE']),)
        fused_tile_exp_kernel[grid](
            output_flat,
            input_flat,
            input_numel,
            output_numel,
            BLOCK_SIZE=1024,
        )
        
        ctx.save_for_backward(input, output)
        ctx.adjusted_dims = adjusted_dims
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        adjusted_dims = ctx.adjusted_dims
        
        # Gradient of exp(y) is exp(y) * grad_output
        grad_input = torch.zeros_like(input)
        grad_output_flat = grad_output.contiguous().view(-1)
        grad_input_flat = grad_input.view(-1)
        input_numel = grad_input_flat.numel()
        output_numel = grad_output_flat.numel()
        
        # Launch the gradient kernel
        grid = lambda meta: (triton.cdiv(output_numel, meta['BLOCK_SIZE']),)
        fused_tile_exp_grad_kernel[grid](
            grad_input_flat,
            grad_output_flat,
            output.view(-1),
            input_numel,
            output_numel,
            BLOCK_SIZE=1024,
        )
        
        return grad_input, None

@triton.jit
def fused_tile_exp_kernel(
    output_ptr,
    input_ptr,
    input_numel,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for idx in range(pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE):
        if idx >= output_numel:
            return
        # Compute input index by modulo input_numel
        input_idx = idx % input_numel
        # Load input value
        input_val = tl.load(input_ptr + input_idx)
        # Compute exponential
        output_val = tl.exp(input_val)
        # Store result
        tl.store(output_ptr + idx, output_val)

@triton.jit
def fused_tile_exp_grad_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    input_numel,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for idx in range(pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE):
        if idx >= output_numel:
            return
        input_idx = idx % input_numel
        # Load output value (exp(input_val))
        output_val = tl.load(output_ptr + idx)
        # Load grad_output value
        grad_output_val = tl.load(grad_output_ptr + idx)
        # Compute gradient: exp(input_val) * grad_output_val
        grad = output_val * grad_output_val
        # Accumulate gradients to the corresponding input index
        tl.atomic_add(grad_input_ptr + input_idx, grad)

def fused_tile_exp(input, dims, *, out=None):
    result = FusedTileExpAutograd.apply(input, dims)
    if out is not None:
        out.copy_(result)
        return out
    return result
