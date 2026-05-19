import torch
import triton
import triton.language as tl

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Check shapes compatibility
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert vec.dim() == 1, "Vector must be a 1D tensor"
    assert input.size(1) == vec.size(0), "Incompatible dimensions for matrix-vector multiplication"

    # Broadcast other to match the shape of the output from sigmoid
    if isinstance(other, torch.Tensor):
        other = other.view((1,))

    # Determine the device and dtype
    device = input.device
    dtype = input.dtype

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Convert inputs to Triton tensors
    A = triton.In(input.contiguous(), requires_grad=input.requires_grad)
    v = triton.In(vec.contiguous(), requires_grad=vec.requires_grad)
    z = triton.Out(torch.zeros_like(input), requires_grad=True)
    s = triton.Out(torch.zeros_like(input), requires_grad=True)
    y = triton.Out(out, requires_grad=out.requires_grad)

    # Launch the kernel
    num_warps = 4
    grid = (triton.cdiv(input.shape[0], 32), 1)
    block = (32, 1, 1)
    fused_mv_sigmoid_sub_kernel[grid, block](A.data_ptr(), v.data_ptr(), z.data_ptr(), s.data_ptr(), y.data_ptr(),
                                            input.shape[0], input.shape[1], alpha, other.item(),
                                            BLOCK_SIZE_M=32, BLOCK_SIZE_N=1, num_warps=num_warps)

    return y
