import triton
import triton.language as tl
import torch

# Utility function to determine if int32 indices can be used
def can_use_int32_index(num_elements):
    return num_elements <= 2**31 - 1

# Kernel for the first stage of reduction
@triton.jit
def argmax_kernel_1(inp_ptr, mid_value_ptr, mid_index_ptr, M, BLOCK_SIZE, INT64_INDEX, **meta):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    inp = tl.load(inp_ptr + offsets, mask=mask, other=-float('inf'))
    
    max_value = tl.max(inp, axis=0)
    max_index = tl.argmax(inp, axis=0)
    
    tl.store(mid_value_ptr + pid, max_value)
    if INT64_INDEX:
        tl.store(mid_index_ptr + pid, max_index.to(tl.int64))
    else:
        tl.store(mid_index_ptr + pid, max_index.to(tl.int32))

# Kernel for the second stage of reduction
@triton.jit
def argmax_kernel_2(mid_value_ptr, mid_index_ptr, out_ptr, mid_size, BLOCK_MID, **meta):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_MID + tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    mid_value = tl.load(mid_value_ptr + offsets, mask=mask, other=-float('inf'))
    mid_index = tl.load(mid_index_ptr + offsets, mask=mask, other=0)
    
    max_value = tl.max(mid_value, axis=0)
    max_index = tl.argmax(mid_value, axis=0)
    
    tl.store(out_ptr, mid_index[max_index])

# Kernel for dimension-specific reduction
@triton.jit
def argmax_kernel(inp_ptr, out_ptr, M, N, K, BLOCK_M, BLOCK_N, **meta):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    inp = tl.load(inp_ptr + offsets_m[:, None] * N + offsets_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))
    
    max_value = tl.max(inp, axis=1)
    max_index = tl.argmax(inp, axis=1)
    
    tl.store(out_ptr + offsets_m, max_index)

# Wrapper function for argmax
def argmax(input_tensor, dim=None):
    if dim is None:
        # Flatten the tensor and perform two-stage reduction
        input_tensor = input_tensor.flatten()
        M = input_tensor.numel()
        BLOCK_SIZE = 1024  # Define block size for stage 1
        mid_size = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        INT64_INDEX = not can_use_int32_index(M)
        
        mid_value = torch.empty(mid_size, dtype=input_tensor.dtype, device=input_tensor.device)
        mid_index = torch.empty(mid_size, dtype=torch.int64 if INT64_INDEX else torch.int32, device=input_tensor.device)
        
        grid = (mid_size,)
        argmax_kernel_1[grid](input_tensor, mid_value, mid_index, M, BLOCK_SIZE, INT64_INDEX)
        
        out = torch.empty(1, dtype=torch.int64, device=input_tensor.device)
        BLOCK_MID = 1024  # Define block size for stage 2
        grid_mid = (1,)
        argmax_kernel_2[grid_mid](mid_value, mid_index, out, mid_size, BLOCK_MID)
        
        return out.item()
    else:
        # Perform reduction along the specified dimension
        shape = input_tensor.shape
        M = int(torch.prod(torch.tensor(shape[:dim])))
        N = shape[dim]
        K = int(torch.prod(torch.tensor(shape[dim+1:])))
        
        BLOCK_M = 64  # Define block size for M
        BLOCK_N = 128  # Define block size for N
        
        out = torch.empty((M, K), dtype=torch.int64, device=input_tensor.device)
        
        grid = (M // BLOCK_M, K // BLOCK_N)
        argmax_kernel[grid](input_tensor, out, M, N, K, BLOCK_M, BLOCK_N)
        
        return out

# Example usage
input_tensor = torch.randn(256, 256, device='cuda')
result = argmax(input_tensor, dim=1)
print(result)
