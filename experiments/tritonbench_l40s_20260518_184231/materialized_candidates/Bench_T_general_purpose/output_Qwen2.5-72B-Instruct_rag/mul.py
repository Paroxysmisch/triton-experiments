import triton
import triton.language as tl
import torch

# Kernel for tensor-tensor multiplication with broadcasting
@triton.jit
def mul_tensor_kernel(A_ptr, B_ptr, C_ptr, stride_a0, stride_a1, stride_b0, stride_b1, stride_c0, stride_c1, n0, n1, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of two tensors A and B, and store the result in C.
    
    Parameters:
    - A_ptr: Pointer to the tensor A (input).
    - B_ptr: Pointer to the tensor B (input).
    - C_ptr: Pointer to the tensor C (output).
    - stride_a0, stride_a1: Strides for tensor A.
    - stride_b0, stride_b1: Strides for tensor B.
    - stride_c0, stride_c1: Strides for tensor C.
    - n0, n1: Dimensions of the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n0 * n1
    arange_0 = tl.arange(0, n0)
    arange_1 = tl.arange(0, n1)
    A = tl.load(A_ptr + arange_0[:, None] * stride_a0 + arange_1[None, :] * stride_a1, mask=mask)
    B = tl.load(B_ptr + arange_0[:, None] * stride_b0 + arange_1[None, :] * stride_b1, mask=mask)
    C = A * B
    tl.store(C_ptr + arange_0[:, None] * stride_c0 + arange_1[None, :] * stride_c1, C, mask=mask)

# Kernel for tensor-scalar multiplication
@triton.jit
def mul_scalar_kernel(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of a tensor A and a scalar B, and store the result in C.
    
    Parameters:
    - A_ptr: Pointer to the tensor A (input).
    - B: Scalar value to multiply with.
    - C_ptr: Pointer to the tensor C (output).
    - n_elements: Total number of elements in the tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A * B
    tl.store(C_ptr + offsets, C, mask=mask)

# Wrapper function to handle the multiplication
def mul(input, other, *, out=None):
    """
    Multiplies the input tensor by another tensor or a number, supporting broadcasting to a common shape, type promotion, and integer, float, and complex inputs.
    
    Parameters:
    - input (Tensor): The input tensor.
    - other (Tensor or Number): The tensor or number to multiply the input by.
    - out (Tensor, optional): The output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the multiplication.
    """
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    if isinstance(other, torch.Tensor):
        # Ensure both tensors have the same shape or can be broadcasted
        input, other, out = torch.broadcast_tensors(input, other, out)
        
        # Get the strides and dimensions
        stride_a0, stride_a1 = input.stride()
        stride_b0, stride_b1 = other.stride()
        stride_c0, stride_c1 = out.stride()
        n0, n1 = input.shape
        
        # Determine the block size
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(input.numel())))
        grid_size = triton.cdiv(input.numel(), block_size)
        
        # Launch the kernel
        mul_tensor_kernel[(grid_size, 1, 1)](input, other, out, stride_a0, stride_a1, stride_b0, stride_b1, stride_c0, stride_c1, n0, n1, BLOCK_SIZE=block_size)
    else:
        # Scalar multiplication
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(input.numel())))
        grid_size = triton.cdiv(input.numel(), block_size)
        
        # Launch the kernel
        mul_scalar_kernel[(grid_size, 1, 1)](input, other, out, input.numel(), BLOCK_SIZE=block_size)
    
    return out
