import torch
import triton
import triton.language as tl

class GridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
        output = torch.empty_like(input)
        input_grad = torch.empty_like(input)
        grid_grad = torch.empty_like(grid)

        if grid.dtype == torch.int64:
            grid = grid.to(torch.float32)

        if align_corners:
            grid = grid * 2.0
            if grid.dim() == 4:
                grid[:, :, :, 0] = grid[:, :, :, 0] * (input.shape[2] - 1)
                grid[:, :, :, 1] = grid[:, :, :, 1] * (input.shape[3] - 1)
            else:
                grid[:, :, :, :, 0] = grid[:, :, :, :, 0] * (input.shape[3] - 1)
                grid[:, :, :, :, 1] = grid[:, :, :, :, 1] * (input.shape[4] - 1)

        if padding_mode == "zeros":
            output_tensor = triton_grid_sample(input, grid, output, 0, 0, mode,
                                                input.shape[2], input.shape[3],
                                                input.shape[4] if input.dim() == 5 else 1,
                                                output.stride(0), output.stride(1),
                                                output.stride(2), output.stride(3),
                                                output.stride(4) if output.dim() == 5 else 1)
        elif padding_mode == "border":
            output_tensor = triton_grid_sample(input, grid, output,
                                                -input.shape[2] if input.stride(2) > 0 else input.shape[2] - 1,
                                                -input.shape[3] if input.stride(3) > 0 else input.shape[3] - 1,
                                                mode, input.shape[2], input.shape[3],
                                                input.shape[4] if input.dim() == 5 else 1,
                                                output.stride(0), output.stride(1),
                                                output.stride(2), output.stride(3),
                                                output.stride(4) if output.dim() == 5 else 1)
        else:
            output_tensor = triton_grid_sample(input, grid, output,
                                                -input.shape[2] + 1, -input.shape[3] + 1,
                                                mode, input.shape[2], input.shape[3],
                                                input.shape[4] if input.dim() == 5 else 1,
                                                output.stride(0), output.stride(1),
                                                output.stride(2), output.stride(3),
                                                output.stride(4) if output.dim() == 5 else 1)

        ctx.save_for_backward(input, grid)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.input_shape = input.shape
        ctx.grid_shape = grid.shape
        ctx.output_shape = output_tensor.shape

        return output_tensor

    @staticmethod
    def backward(ctx, out_grad):
        (input, grid) = ctx.saved_tensors
        (input_grad, grid_grad) = (torch.empty_like(input), torch.empty_like(grid))

        if grid.dtype == torch.int64:
            grid = grid.to(torch.float32)

        if ctx.align_corners:
            grid = grid * 2.0
            if grid.dim() == 4:
                grid[:, :, :, 0] = grid[:, :, :, 0] * (input.shape[2] - 1)
                grid[:, :, :, 1] = grid[:, :, :, 1] * (input.shape[3] - 1)
            else:
                grid[:, :, :, :, 0] = grid[:, :, :, :, 0] * (input.shape[3] - 1)
                grid[:, :, :, :, 1] = grid[:, :, :, :, 1] * (input.shape[4] - 1)

        triton_grid_sample_backward(input, grid, out_grad, input_grad, grid_grad,
                                     ctx.mode, ctx.padding_mode, ctx.align_corners,
                                     ctx.input_shape, ctx.grid_shape,
                                     input.stride(0), input.stride(1),
                                     input.stride(2), input.stride(3),
                                     input.stride(4) if input.dim() == 5 else 1,
                                     grid.stride(0), grid.stride(1),
                                     grid.stride(2), grid.stride(3),
                                     grid.stride(4) if grid.dim() == 5 else 1,
                                     out_grad.stride(0), out_grad.stride(1),
                                     out_grad.stride(2), out_grad.stride(3),
                                     out_grad.stride(4) if out_grad.dim() == 5 else 1)

        return input_grad, grid_grad, None, None, None, None

@triton.jit
def triton_grid_sample(input, grid, output, xmin, ymin, mode, width, height,
                       channels, out_nrows, out_ncols, out_nchannels, out_hstride,
                       out_wstride, out_cstride):
    pidz = tl.program_id(axis=0)
    pidj = tl.program_id(axis=1)
    pids = tl.program_id(axis=2)
    num_z = tl.num_programs(axis=0)
    num_j = tl.num_programs(axis=1)
    num_samples = tl.num_programs(axis=2)

    if (num_z > 1 or num_j > 1 or num_samples > 1):
        grid_idx = pidz * num_j * num_samples + pidj * num_samples + pids
        output_idx = grid_idx
        input_idx = grid_idx
        grid += grid_idx * grid.strides[0]
        output += output_idx * out_nrows * out_ncols * out_nchannels
        input += input_idx * width * height * channels
    else:
        output_idx = tl.arange(0, out_nrows * out_ncols * out_nchannels)
        grid_idx = tl.zeros([out_nrows * out_ncols * out_nchannels], dtype=tl.int32)
        input_idx = tl.zeros([out_nrows * out_nchannels * out_ncols], dtype=tl.int32)

    grid_offset = tl.zeros([out_nrows * out_ncols * out_nchannels], dtype=grid.dtype.element_ty)
    grid_offset = tl.where(output_idx < (out_nrows * out_ncols * out_nchannels), output_idx, grid_offset)
