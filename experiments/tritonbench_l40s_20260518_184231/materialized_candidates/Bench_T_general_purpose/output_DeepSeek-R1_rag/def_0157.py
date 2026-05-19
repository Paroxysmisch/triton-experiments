import triton
import triton.language as tl
import torch
import math

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    
    # Determine the integer type and shift based on input dtype
    dtype = input_ptr.dtype
    if dtype == tl.float32:
        input_int = tl.bitcast(input_val, tl.int32)
        shift = 31
    elif dtype == tl.float16 or dtype == tl.bfloat16:
        input_int = tl.bitcast(input_val, tl.int16)
        shift = 15
    elif dtype == tl.float64:
        input_int = tl.bitcast(input_val, tl.int64)
        shift = 63
    else:
        tl.static_assert(False, "Unsupported dtype for signbit kernel")
    
    sign_bit = (input_int >> shift) & 1
    output = sign_bit.to(tl.int1)
    tl.store(output_ptr + offsets, output, mask=mask)

def compute_signbit(input: torch.Tensor) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("Input tensor must be a float type for signbit computation.")
    output = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    block_size = triton.next_power_of_2(1024)
    grid_size = triton.cdiv(n_elements, block_size)
    signbit_kernel[(grid_size, 1, 1)](input, output, n_elements, BLOCK_SIZE=block_size)
    return output

@triton.jit
def bitwise_and_kernel(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    c = a & b
    tl.store(C_ptr + offsets, c, mask=mask)

def bitwise_and_tensor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == B.dtype, "Tensors must have the same dtype for bitwise AND."
    assert A.shape == B.shape, "Tensors must have the same shape for bitwise AND."
    C = torch.empty_like(A)
    n_elements = C.numel()
    if n_elements == 0:
        return C
    block_size = 1024
    grid_size = triton.cdiv(n_elements, block_size)
    bitwise_and_kernel[(grid_size, 1, 1)](A, B, C, n_elements, BLOCK_SIZE=block_size)
    return C

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not (other.dtype.is_integer or other.dtype == torch.bool):
        raise ValueError("other tensor must be of integral or boolean types for bitwise AND.")
    
    signbit_result = compute_signbit(input)
    input_cast = input.to(other.dtype)
    bitwise_and_result = bitwise_and_tensor(input_cast, other)
    
    return signbit_result, bitwise_and_result
