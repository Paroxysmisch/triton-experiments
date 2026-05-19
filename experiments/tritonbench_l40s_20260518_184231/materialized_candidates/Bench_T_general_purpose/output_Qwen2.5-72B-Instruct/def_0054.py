import triton
import triton.language as tl

@triton.jit
def gelu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    stride, padding, dilation, groups,
    approximate,
    BLOCK_SIZE: tl.constexpr
):
    # Extract shapes and strides
    N, IC, IH, IW = input_shape
    OC, KIC, KH, KW = weight_shape
    OH, OW = output_shape[-2:]
    
    # Compute the output size
    IC_per_group = IC // groups
    OC_per_group = OC // groups
    
    # Compute the grid and block indices
    pid = tl.program_id(axis=0)
    num_blocks = (OH * OW * OC) // BLOCK_SIZE
    block_id = pid % num_blocks
    block_start = block_id * BLOCK_SIZE
    
    # Compute the output coordinates
    oc = (block_start // (OH * OW)) % OC
    oh = (block_start // OW) % OH
    ow = block_start % OW
    
    # Initialize the output value
    out_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute the input and weight coordinates
    for icg in range(IC_per_group):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride[0] - padding[0] + kh * dilation[0]
                iw = ow * stride[1] - padding[1] + kw * dilation[1]
                
                # Check if the coordinates are within bounds
                if ih >= 0 and ih < IH and iw >= 0 and iw < IW:
                    input_offset = (pid // num_blocks) * IC * IH * IW + icg * IH * IW + ih * IW + iw
                    weight_offset = oc * KIC * KH * KW + icg * KH * KW + kh * KW + kw
                    input_val = tl.load(input_ptr + input_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)
                    out_val += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_offset = oc
        bias_val = tl.load(bias_ptr + bias_offset)
        out_val += bias_val
    
    # Apply GELU activation
    if approximate == 'none':
        out_val = out_val * tl.math.cdf(out_val)
    elif approximate == 'tanh':
        out_val = 0.5 * out_val * (1 + tl.math.tanh(0.7978845608028654 * (out_val + 0.044715 * out_val * out_val * out_val)))
    
    # Store the result
    output_offset = (pid // num_blocks) * OC * OH * OW + oc * OH * OW + oh * OW + ow
    tl.store(output_ptr + output_offset, out_val)

import torch
import triton
import triton.language as tl

def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, 
                dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, 
                approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Convert inputs to appropriate types
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Compute the output shape
    N, IC, IH, IW = input.shape
    OC, KIC, KH, KW = weight.shape
    if bias is not None:
        assert bias.shape[0] == OC, "Bias shape mismatch"
    
    OH = (IH + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (IW + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((N, OC, OH, OW), device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    grid = (N * OC * OH * OW // 1024 + 1, 1, 1)
    gelu_conv2d_kernel[grid](
        input, weight, bias, out,
        (N, IC, IH, IW), (OC, KIC, KH, KW), (N, OC, OH, OW),
        stride, padding, dilation, groups,
        approximate,
        BLOCK_SIZE=1024
    )
    
    return out
