import triton
import triton.language as tl
from torch.fft import ifftshift

@triton.jit
def example_function(input_tensor):
    # Use ifftshift function
    shifted_tensor = ifftshift(input_tensor)
    # Rest of your code
