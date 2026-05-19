import torch
import triton

# Create input tensor
input_tensor = torch.tensor([1.0, 4.0, 9.0], device='cuda')

# Create output tensor
output_tensor = torch.empty_like(input_tensor, device='cuda')

# Call rsqrt function
triton.ops.rsqrt(input_tensor, output_tensor)

# Print output tensor
print(output_tensor)
