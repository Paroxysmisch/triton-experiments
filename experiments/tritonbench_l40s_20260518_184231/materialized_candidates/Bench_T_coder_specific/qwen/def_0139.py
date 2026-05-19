import triton
import triton.language as tl
from typing import Optional

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor:
    T = input.dtype
    F = tl.float32 if T == tl.int32 else T

    if dim is None:
        dim = list(range(input.ndim))

    if isinstance(dim, int):
        dim = [dim]

    # Determine the shape of the output tensor
    output_shape = list(input.shape)
    for d in dim:
        output_shape[d] = 1 if keepdim else 1

    # Create output tensors
    if out is None:
        out = torch.zeros(output_shape, dtype=T, device=input.device)

    # Initialize mean and variance tensors
    mean = torch.zeros_like(out, dtype=F)
    var = torch.zeros_like(out, dtype=F)

    # Define strides
    stride_K = [input.stride(d) for d in dim]
    stride_L = [output_shape[d] for d in dim]
    stride_M = [mean.stride(d) for d in dim]
    stride_N = input.numel()
    stride_O = out.numel()

    # Launch the kernel
    num_blocks = math.ceil(N / block_size)
    grid_size = (num_blocks,)
    std_kernel[input.dtype](X=input.data_ptr(), Y=out.data_ptr(), M=mean.data_ptr(), V=var.data_ptr(),
                          N=N, C=correction, K=len(dim), stride_K=stride_K, stride_L=stride_L,
                          stride_M=stride_M, stride_N=stride_N, stride_O=stride_O,
                          block=(block_size,), grid=grid_size)

    return out
