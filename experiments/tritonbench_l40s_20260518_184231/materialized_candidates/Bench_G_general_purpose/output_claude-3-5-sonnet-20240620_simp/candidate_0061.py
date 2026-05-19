import triton
import triton.language as tl
import torch

@triton.jit
def _sgmv_expand_slice_kernel(
    # Pointers to matrices
    output_ptr,          # Pointer to output matrix
    input_ptr,          # Pointer to input matrix
    weight_ptr,         # Pointer to weight matrix
    lora_weight_ptr,    # Pointer to LoRA weight matrix
    # Matrix dimensions
    batch_size,         # Batch size
    in_features,        # Input feature dimension
    out_features,       # Output feature dimension
    # Additional parameters
    input_stride,       # Stride for input matrix
    output_stride,      # Stride for output matrix
    weight_stride,      # Stride for weight matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for computation
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and feature indices
    batch_id = pid // (out_features // BLOCK_SIZE)
    feature_id = (pid % (out_features // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Compute input/output offsets
    input_offset = batch_id * input_stride
    output_offset = batch_id * output_stride + feature_id
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load input block
    x = tl.load(input_ptr + input_offset + tl.arange(0, in_features))
    
    # Main computation loop
    for i in range(0, in_features, BLOCK_SIZE):
        # Load weight block
        w = tl.load(weight_ptr + feature_id * weight_stride + i + tl.arange(0, BLOCK_SIZE))
        
        # Load LoRA weight block if provided
        if lora_weight_ptr is not None:
            lw = tl.load(lora_weight_ptr + feature_id * weight_stride + i + tl.arange(0, BLOCK_SIZE))
            w = w + lw
            
        # Compute partial dot product
        acc += tl.dot(x[i:i + BLOCK_SIZE], w)
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

def _sgmv_expand_slice(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    lora_weight: torch.Tensor = None,
    output_size: int = None
) -> torch.Tensor:
    """
    Wrapper function for the SGMV kernel with LoRA support
    
    Args:
        input_tensor: Input tensor of shape [batch_size, in_features]
        weight: Weight matrix of shape [out_features, in_features]
        lora_weight: Optional LoRA weight matrix of same shape as weight
        output_size: Optional output size (defaults to weight.shape[0])
    
    Returns:
        Output tensor of shape [batch_size, out_features]
    """
    # Get dimensions
    batch_size = input_tensor.shape[0]
    in_features = input_tensor.shape[1]
    out_features = output_size if output_size is not None else weight.shape[0]
    
    # Compute optimal block size
    BLOCK_SIZE = triton.next_power_of_2(min(128, in_features))
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), 
                        device=input_tensor.device, 
                        dtype=input_tensor.dtype)
    
    # Prepare grid
    grid = (batch_size * (out_features // BLOCK_SIZE),)
    
    # Launch kernel
    _sgmv_expand_slice_kernel[grid](
        output,
        input_tensor,
        weight,
        lora_weight,
        batch_size,
        in_features,
        out_features,
        input_tensor.stride(0),
        output.stride(0),
        weight.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
