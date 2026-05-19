import torch
import numpy as np

# Original function
def original_function(input: torch.Tensor):
    return torch.cos(input).mean(dim=1)

# Triton wrapper function
from your_triton_wrapper_module import cos_avg_pool1d

# Create a random input tensor
input = torch.from_numpy(np.random.rand(10, 32).astype(np.float32))

# Compute outputs
original_output = original_function(input)
triton_output = cos_avg_pool1d(input, kernel_size=1)

# Compare outputs
np.testing.assert_allclose(original_output.numpy(), triton_output.numpy(), atol=1e-6)
