import triton
from triton import csrc

# Load the Triton kernel module
module = csrc.load_module("ldl_factor_kernel.so")

# Define the Triton wrapper function
@triton.jit()
def linalg_ldl_factor(A, hermitian=False, out=None):
    # Get the shape of the input tensor
    batch_size = A.shape[:-2]
    n = A.shape[-1]

    # Allocate memory for the output tensors
    LD = triton.empty_like(A)
    pivots = triton.empty_like(A, dtype=torch.int32)

    # Call the kernel
    module.ldl_factor_kernel(A, n, hermitian, LD, pivots)

    # Return the result as a named tuple
    return LD, pivots
