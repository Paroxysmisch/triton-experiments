import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    # Pointers to tensors
    input_ptr, weight_ptr, output_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels, in_h, in_w, 
    out_h, out_w, kernel_h, kernel_w,
    # Strides for memory access
    stride_n, stride_c, stride_h, stride_w,
    w_stride_o, w_stride_i, w_stride_h, w_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    # Parameters
    padding_h, padding_w, stride_height, stride_width,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    n = pid // (out_channels * out_h * out_w)
    tmp = pid % (out_channels * out_h * out_w)
    oc = tmp // (out_h * out_w)
    tmp = tmp % (out_h * out_w)
    oh = tmp // out_w
    ow = tmp % out_w
    
    # Calculate input window position
    ih_start = oh * stride_height - padding_h
    iw_start = ow * stride_width - padding_w
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(in_channels):
        for kh in range(kernel_h):
            ih = ih_start + kh
            if 0 <= ih < in_h:
                for kw in range(kernel_w):
                    iw = iw_start + kw
                    if 0 <= iw < in_w:
                        # Load input value
                        in_idx = n * stride_n + ic * stride_c + ih * stride_h + iw * stride_w
                        in_val = tl.load(input_ptr + in_idx)
                        
                        # Load weight value
                        w_idx = (oc * w_stride_o + ic * w_stride_i + 
                               kh * w_stride_h + kw * w_stride_w)
                        w_val = tl.load(weight_ptr + w_idx)
                        
                        # Accumulate product
                        acc += in_val * w_val
    
    # Store output
    out_idx = (n * out_stride_n + oc * out_stride_c + 
               oh * out_stride_h + ow * out_stride_w)
    tl.store(output_ptr + out_idx, acc)

def conv2d_forward(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0)):
    # Get dimensions
    batch_size, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    
    # Calculate output dimensions
    out_h = ((in_h + 2 * padding[0] - kernel_h) // stride[0]) + 1
    out_w = ((in_w + 2 * padding[1] - kernel_w) // stride[1]) + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), 
                        device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Get tensor strides
    stride_n, stride_c, stride_h, stride_w = input_tensor.stride()
    w_stride_o, w_stride_i, w_stride_h, w_stride_w = weight_tensor.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid = (batch_size * out_channels * out_h * out_w,)
    
    # Launch kernel
    conv2d_forward_kernel[grid](
        input_tensor, weight_tensor, output,
        batch_size, in_channels, out_channels, in_h, in_w,
        out_h, out_w, kernel_h, kernel_w,
        stride_n, stride_c, stride_h, stride_w,
        w_stride_o, w_stride_i, w_stride_h, w_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        padding[0], padding[1], stride[0], stride[1],
        BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    return output
