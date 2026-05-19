import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    # Input, Weight, Output Pointers
    input_ptr, weight_ptr, output_ptr,
    # Input Tensor Metadata
    input_batch, input_channels, input_h, input_w,
    input_stride_b, input_stride_ic, input_stride_h, input_stride_w,
    # Weight Tensor Metadata
    weight_oc, weight_ic, weight_kh, weight_kw,
    weight_stride_oc, weight_stride_ic, weight_stride_h, weight_stride_w,
    # Output Tensor Metadata
    output_batch, output_oc, output_h, output_w,
    output_stride_b, output_stride_oc, output_stride_h, output_stride_w,
    # Convolution Parameters
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    groups,
    # Data Type Configuration
    fp16_enabled: tl.constexpr,
    tf32_enabled: tl.constexpr,
    # Block Sizes
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Determine program IDs and block offsets
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh_block = tl.program_id(2)
    pid_ow_block = tl.program_id(3)

    # Calculate block ranges
    batch_start = pid_batch * BLOCK_SIZE_BATCH
    oc_start = pid_oc * BLOCK_SIZE_OUT_FEAT
    oh_start = pid_oh_block * BLOCK_SIZE_H
    ow_start = pid_ow_block * BLOCK_SIZE_W

    # Clamp ranges to tensor dimensions
    batch_end = tl.minimum((pid_batch + 1) * BLOCK_SIZE_BATCH, input_batch)
    oc_end = tl.minimum((pid_oc + 1) * BLOCK_SIZE_OUT_FEAT, weight_oc)
    oh_end = tl.minimum((pid_oh_block + 1) * BLOCK_SIZE_H, output_h)
    ow_end = tl.minimum((pid_ow_block + 1) * BLOCK_SIZE_W, output_w)

    # Determine group parameters
    group_size_oc = weight_oc // groups
    group_id = oc_start // group_size_oc
    ic_start = group_id * (input_channels // groups)
    ic_end = (group_id + 1) * (input_channels // groups)

    # Data type handling
    if fp16_enabled:
        input_dtype = tl.float16
        weight_dtype = tl.float16
    else:
        input_dtype = tl.float32
        weight_dtype = tl.float32

    # Main computation loop
    for n in range(batch_start, batch_end):
        for oc in range(oc_start, oc_end):
            for oh in range(oh_start, oh_end):
                for ow in range(ow_start, ow_end):
                    acc = tl.zeros((1,), dtype=tl.float32)
                    for ic in range(ic_start, ic_end):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # Calculate input positions
                                ih = oh * stride_h - padding_h + kh
                                iw = ow * stride_w - padding_w + kw
                                
                                if ih >= 0 and ih < input_h and iw >= 0 and iw < input_w:
                                    # Load input and weight values
                                    input_val = tl.load(
                                        input_ptr + n * input_stride_b + 
                                        ic * input_stride_ic + 
                                        ih * input_stride_h + 
                                        iw * input_stride_w,
                                        dtype=input_dtype
                                    )
                                    weight_val = tl.load(
                                        weight_ptr + oc * weight_stride_oc + 
                                        ic * weight_stride_ic + 
                                        kh * weight_stride_h + 
                                        kw * weight_stride_w,
                                        dtype=weight_dtype
                                    )
                                    acc += tl.cast(input_val * weight_val, tl.float32)
                    
                    # Store result
                    output_idx = (
                        n * output_stride_b +
                        oc * output_stride_oc +
                        oh * output_stride_h +
                        ow * output_stride_w
                    )
                    tl.store(output_ptr + output_idx, acc)

def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    groups: int,
    fp16: bool = False,
    tf32: bool = False,
):
    # Validate inputs
    assert input.ndim == 4, "Input must be 4D (N, C, H, W)"
    assert weight.ndim == 4, "Weight must be 4D (OC, IC, KH, KW)"
    
    # Compute output dimensions
    N, C, H, W = input.shape
    OC, IC, KH, KW = weight.shape
    out_h = (H + 2*padding - KH) // stride + 1
    out_w = (W + 2*padding - KW) // stride + 1
    
    # Initialize output tensor
    output = torch.empty((N, OC, out_h, out_w), device=input.device, dtype=input.dtype)
    
    # Configure block sizes
    BLOCK_B = 4
    BLOCK_IC = 16
    BLOCK_OC = 32
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Compute grid dimensions
    grid_b = triton.cdiv(N, BLOCK_B)
    grid_oc = triton.cdiv(OC, BLOCK_OC)
    grid_oh = triton.cdiv(out_h, BLOCK_H)
    grid_ow = triton.cdiv(out_w, BLOCK_W)
    
    # Launch kernel
    conv2d_forward_kernel[ (grid_b, grid_oc, grid_oh, grid_ow) ](
        input, weight, output,
        # Input metadata
        N, C, H, W,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        # Weight metadata
        OC, IC, KH, KW,
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        # Output metadata
        N, OC, out_h, out_w,
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Convolution parameters
        KH, KW, stride, stride, padding, padding, groups,
        # Data type flags
        fp16, tf32,
        # Block sizes
        BLOCK_B, BLOCK_IC, BLOCK_OC, BLOCK_H, BLOCK_W
    )
    
    return output
