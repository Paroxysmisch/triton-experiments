import torch
import triton
import triton.language as tl

def adjust_dims(dims, ndim):
    adjusted = list(dims)
    while len(adjusted) < ndim:
        adjusted = [1] + adjusted
    return tuple(adjusted)

@triton.jit
def fused_tile_exp_kernel(
    input_ptr,
    output_ptr,
    input_shape_ptr,
    tile_dims_ptr,
    input_strides_ptr,
    output_strides_ptr,
    ndim,
    input_numel,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for idx in range(pid * BLOCK_SIZE, output_numel, BLOCK_SIZE * tl.num_programs(0)):
        if idx >= output_numel:
            return
        input_idx = 0
        remaining = idx
        for dim in range(ndim):
            output_stride = tl.load(output_strides_ptr + dim)
            input_size = tl.load(input_shape_ptr + dim)
            tile_dim = tl.load(tile_dims_ptr + dim)
            output_dim_size = input_size * tile_dim
            coord_out = (remaining // output_stride) % output_dim_size
            coord_in = coord_out // tile_dim
            input_stride = tl.load(input_strides_ptr + dim)
            input_idx += coord_in * input_stride
            remaining = remaining % output_stride
        x = tl.load(input_ptr + input_idx)
        y = tl.exp(x)
        tl.store(output_ptr + idx, y)

class FusedTileExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, adjusted_dims):
        ctx.save_for_backward(input, torch.tensor(adjusted_dims, dtype=torch.int))
        output_shape = tuple(s * d for s, d in zip(input.shape, adjusted_dims))
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        input_flat = input.view(-1)
        output_flat = output.view(-1)
        input_shape = input.shape
        input_strides = list(input.stride())
        output_strides = list(output.stride())
        input_shape_tensor = torch.tensor(input_shape, dtype=torch.int, device=input.device)
        tile_dims_tensor = torch.tensor(adjusted_dims, dtype=torch.int, device=input.device)
        input_strides_tensor = torch.tensor(input_strides, dtype=torch.int, device=input.device)
        output_strides_tensor = torch.tensor(output_strides, dtype=torch.int, device=input.device)
        ndim = input.ndim
        input_numel = input_flat.numel()
        output_numel = output_flat.numel()
        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(output_numel, BLOCK_SIZE),)
        fused_tile_exp_kernel[grid](
            input_flat, output_flat,
            input_shape_tensor, tile_dims_tensor,
            input_strides_tensor, output_strides_tensor,
            ndim, input_numel, output_numel,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, adjusted_dims_tensor = ctx.saved_tensors
        adjusted_dims = adjusted_dims_tensor.tolist()
        exp_input = torch.exp(input)
        new_shape = []
        for i in range(input.ndim):
            new_shape.append(input.size(i))
            new_shape.append(adjusted_dims[i])
        grad_reshaped = grad_output.view(new_shape)
        sum_dims = [2 * i + 1 for i in range(input.ndim)]
        grad_sum = grad_reshaped.sum(dim=sum_dims)
        grad_input = grad_sum * exp_input
        return grad_input, None

def fused_tile_exp(input, dims, *, out=None):
    dims = tuple(dims)
    adjusted_dims = adjust_dims(dims, input.ndim)
    result = FusedTileExpFunction.apply(input, adjusted_dims)
    if out is not None:
        out.copy_(result)
        return out
    return result
