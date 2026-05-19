{{ code }}
@triton.jit
def mul_sub_kernel(input_ptr, other_mul_ptr, other_sub_ptr, out_ptr, alpha, N):
    # Compute the index for each thread
    idx = triton.program_id(0) * triton.block_size(0) + triton.thread_id(0)
    
    # Ensure we do not go out of bounds
    if idx < N:
        # Load input and other tensors
        input_val = input_ptr[idx]
        other_mul_val = other_mul_ptr[idx] if isinstance(other_mul_ptr, triton.Tensor) else other_mul_ptr
        other_sub_val = other_sub_ptr[idx] if isinstance(other_sub_ptr, triton.Tensor) else other_sub_ptr
        
        # Perform the operation
        out_val = (input_val * other_mul_val) - (alpha * other_sub_val)
        
        # Store the result
        out_ptr[idx] = out_val

{{ code }}
def mul_sub(input: Tensor, other_mul: Union[Tensor, float], other_sub: Union[Tensor, float], alpha: float = 1, out: Optional[Tensor] = None) -> Tensor:
    # Determine the size of the input tensor
    N = input.shape[0]
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    mul_sub_kernel[(N + 255) // 256](input, other_mul, other_sub, out, alpha, N)
    
    return out
