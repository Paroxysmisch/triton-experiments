import triton
import triton.language as tl

# Define the SWiGLU forward kernel
@triton.jit
def _swiglu_forward_kernel(
    a_ptr,  # Pointer to input tensor a
    b_ptr,  # Pointer to input tensor b
    c_ptr,  # Pointer to output tensor c
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Create offsets for the block
    mask = offsets < n_elements  # Create a mask to handle the last block

    a = tl.load(a_ptr + offsets, mask=mask)  # Load elements from a
    b = tl.load(b_ptr + offsets, mask=mask)  # Load elements from b

    # Apply the SiLU activation to a
    a_silu = a * tl.sigmoid(a)

    # Compute the element-wise product of a_silu and b
    c = a_silu * b

    # Store the result in c
    tl.store(c_ptr + offsets, c, mask=mask)

# Define the SWiGLU backward kernel
@triton.jit
def _swiglu_backward_kernel(
    a_ptr,  # Pointer to input tensor a
    b_ptr,  # Pointer to input tensor b
    dc_ptr,  # Pointer to gradient of output tensor c
    da_ptr,  # Pointer to gradient of input tensor a
    db_ptr,  # Pointer to gradient of input tensor b
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Create offsets for the block
    mask = offsets < n_elements  # Create a mask to handle the last block

    a = tl.load(a_ptr + offsets, mask=mask)  # Load elements from a
    b = tl.load(b_ptr + offsets, mask=mask)  # Load elements from b
    dc = tl.load(dc_ptr + offsets, mask=mask)  # Load elements from dc

    # Apply the SiLU activation to a
    a_silu = a * tl.sigmoid(a)

    # Compute the gradient with respect to b
    db = dc * a_silu

    # Compute the gradient with respect to a
    da = dc * b * (tl.sigmoid(a) + a * (1 - tl.sigmoid(a)))

    # Store the gradients in da and db
    tl.store(da_ptr + offsets, da, mask=mask)
    tl.store(db_ptr + offsets, db, mask=mask)

# Define the forward function
def swiglu_forward(a, b, c, BLOCK_SIZE=128, num_warps=4):
    n_elements = a.shape[0]
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_forward_kernel[grid](a, b, c, n_elements, BLOCK_SIZE, num_warps=num_warps)

# Define the backward function
def swiglu_backward(a, b, dc, da, db, BLOCK_SIZE=128, num_warps=4):
    n_elements = a.shape[0]
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_backward_kernel[grid](a, b, dc, da, db, n_elements, BLOCK_SIZE, num_warps=num_warps)

import torch

# Create input tensors
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')

# Create output tensor
c = torch.empty_like(a)

# Forward pass
swiglu_forward(a, b, c)

# Create gradient tensor
dc = torch.randn_like(a)

# Create gradient tensors for a and b
da = torch.empty_like(a)
db = torch.empty_like(b)

# Backward pass
swiglu_backward(a, b, dc, da, db)

# Now da and db contain the gradients
print(da)
print(db)
