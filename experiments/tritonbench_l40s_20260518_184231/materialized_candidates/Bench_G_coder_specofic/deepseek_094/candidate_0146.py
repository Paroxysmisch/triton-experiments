import torch
import triton
from triton import Float32, Int32

def fused_add_mul_activation_torch(x, bias, in_, scale=1.0, type=0):
    x_ptr = x.cuda().data_ptr()
    bias_ptr = bias.cuda().data_ptr()
    in_ptr = in_.cuda().data_ptr()
    out_ptr = torch.empty_like(x).cuda().data_ptr()

    size = x.numel()

    grid = lambda : triton.cuda.grid(size, BLOCK_SIZE)
    fused_add_mul_activation_kernel[grid](x_ptr, bias_ptr, in_ptr, out_ptr, size, scale, type)

    return torch.from_numpy(out_ptr)
