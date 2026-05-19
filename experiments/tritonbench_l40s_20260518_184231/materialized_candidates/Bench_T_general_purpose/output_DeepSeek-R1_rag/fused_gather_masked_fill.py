import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    output_ptr,
    input_ptr,
    index_ptr,
    mask_ptr,
    input_row_stride,
    input_col_stride,
    index_row_stride,
    index_col_stride,
    mask_row_stride,
    mask_col_stride,
    output_row_stride,
    output_col_stride,
    n_cols_input,
    n_rows,
    n_cols_output,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_idx = col_offsets

    row_mask = row_idx < n_rows
    col_mask = col_idx < n_cols_output
    mask = row_mask & col_mask

    index_offset = row_idx * index_row_stride + col_idx * index_col_stride
    index_val = tl.load(index_ptr + index_offset, mask=mask, other=0)

    valid_index = (index_val >= 0) & (index_val < n_cols_input)
    input_col = tl.where(valid_index, index_val, 0)

    input_offset = row_idx * input_row_stride + input_col * input_col_stride
    input_val = tl.load(input_ptr + input_offset, mask=mask & valid_index, other=0)

    mask_offset = row_idx * mask_row_stride + col_idx * mask_col_stride
    mask_val = tl.load(mask_ptr + mask_offset, mask=mask, other=False)

    output_val = tl.where(mask_val, value, input_val)
    output_offset = row_idx * output_row_stride + col_idx * output_col_stride
    tl.store(output_ptr + output_offset, output_val, mask=mask)

class FusedGatherMaskedFillFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, index, mask, value, sparse_grad, out):
        ctx.save_for_backward(index, mask)
        ctx.dim = dim
        ctx.sparse_grad = sparse_grad
        ctx.input_shape = input.shape

        input_flattened = input.movedim(dim, -1).flatten(0, -2)
        index_flattened = index.movedim(dim, -1).flatten(0, -2)
        mask_expanded = mask.broadcast_to(index.shape)
        mask_flattened = mask_expanded.movedim(dim, -1).flatten(0, -2)

        output_flattened = torch.empty_like(index_flattened, dtype=input.dtype)
        n_rows, n_cols_output = output_flattened.shape
        n_cols_input = input_flattened.size(1)

        BLOCK_SIZE = triton.next_power_of_2(n_cols_output)
        grid = (n_rows,)

        fused_gather_masked_fill_kernel[grid](
            output_flattened,
            input_flattened,
            index_flattened,
            mask_flattened,
            input_flattened.stride(0),
            input_flattened.stride(1),
            index_flattened.stride(0),
            index_flattened.stride(1),
            mask_flattened.stride(0),
            mask_flattened.stride(1),
            output_flattened.stride(0),
            output_flattened.stride(1),
            n_cols_input,
            n_rows,
            n_cols_output,
            value,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        output = output_flattened.reshape(index.shape)
        if out is not None:
            out.copy_(output)
            output = out
        return output

    @staticmethod
    def backward(ctx, grad_output):
        index, mask = ctx.saved_tensors
        dim = ctx.dim
        sparse_grad = ctx.sparse_grad
        input_shape = ctx.input_shape

        grad_output_masked = grad_output.masked_fill(mask, 0.0)

        grad_input = torch.zeros(input_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_input.scatter_add_(dim, index, grad_output_masked)

        if sparse_grad:
            grad_input = grad_input.to_sparse()

        return grad_input, None, None, None, None, None, None

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    return FusedGatherMaskedFillFunction.apply(input, dim, index, mask, value, sparse_grad, out)
