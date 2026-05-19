import torch
import torch.nn.functional as F

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False
) -> torch.Tensor:
    # Apply ReLU activation
    x = torch.relu(input)
    
    # Validate that exactly one of output_size or output_ratio is provided
    if (output_size is None) == (output_ratio is None):
        raise ValueError("Exactly one of output_size or output_ratio must be specified")
    
    # Apply fractional max pooling
    output, indices = F.fractional_max_pool2d(
        x,
        kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=True
    )
    
    # Return indices if requested
    if return_indices:
        return output, indices
    else:
        return output
