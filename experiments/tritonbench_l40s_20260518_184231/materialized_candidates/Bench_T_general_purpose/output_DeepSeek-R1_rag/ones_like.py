import torch
import triton
import triton.language as tl

@triton.jit
def ones_like_kernel(
    output_ptr,
    n_elements,
    shape_ptr,
    strides_ptr,
    ndim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    indices = tl.zeros((BLOCK_SIZE, ndim), dtype=tl.int32)
    remaining = offsets
    for i in range(ndim):
        stride = tl.load(strides_ptr + i)
        dim_size = tl.load(shape_ptr + i)
        indices = indices * 0  # Reset indices for this dimension (simplified for clarity)
        indices += (remaining // stride).to(tl.int32)[:, None]
        remaining = remaining % stride

    # Compute memory offsets using strides
    memory_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for i in range(ndim):
        stride = tl.load(strides_ptr + i)
        memory_offsets += indices[:, i] * stride

    tl.store(output_ptr + memory_offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Determine output parameters based on input and keyword arguments
    output_dtype = dtype if dtype is not None else input.dtype
    output_layout = layout if layout is not None else input.layout
    output_device = device if device is not None else input.device

    # Create the output tensor with the same properties as input (or overridden)
    out = torch.empty_like(
        input,
        dtype=output_dtype,
        layout=output_layout,
        device=output_device,
        memory_format=memory_format
    )

    # If on CPU, use PyTorch's fill_ for compatibility
    if out.device.type == 'cpu':
        out.fill_(1)
    else:
        # Launch Triton kernel for GPU tensors
        N = out.numel()
        ndim = out.ndim
        shape = torch.tensor(out.shape, device=out.device, dtype=torch.int32)
        strides = torch.tensor(out.stride(), device=out.device, dtype=torch.int32)

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        ones_like_kernel[grid](
            out,
            N,
            shape,
            strides,
            ndim,
            BLOCK_SIZE=1024
        )

    # Set requires_grad attribute
    out.requires_grad = requires_grad

    return out
