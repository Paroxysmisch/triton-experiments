import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    X_ptr,  # Pointer to input tensor X
    Y_ptr,  # Pointer to input tensor Y
    Z_ptr,  # Pointer to output tensor Z
    N,      # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
    ACTIVATION: tl.constexpr  # Type of activation function (e.g., 'RELU', 'SIGMOID')
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the block
    mask = offsets < N  # Mask to handle out-of-bounds accesses

    X = tl.load(X_ptr + offsets, mask=mask)  # Load X
    Y = tl.load(Y_ptr + offsets, mask=mask)  # Load Y

    # Perform fused operation: Z = (X + Y) * (X + Y) with activation
    Z = (X + Y) * (X + Y)
    if ACTIVATION == 'RELU':
        Z = tl.relu(Z)
    elif ACTIVATION == 'SIGMOID':
        Z = tl.sigmoid(Z)

    tl.store(Z_ptr + offsets, Z, mask=mask)  # Store the result back to Z

import torch

def fused_add_mul_activation_torch(X: torch.Tensor, Y: torch.Tensor, activation: str = 'RELU'):
    # Ensure the tensors are on the same device
    assert X.device == Y.device, "Input tensors must be on the same device"
    
    # Ensure the tensors have the same shape
    assert X.shape == Y.shape, "Input tensors must have the same shape"
    
    # Create the output tensor
    Z = torch.empty_like(X)
    
    # Define the grid and block sizes
    N = X.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    fused_add_mul_activation_kernel[(num_blocks,)](
        X, Y, Z, N, BLOCK_SIZE, activation
    )
    
    return Z
