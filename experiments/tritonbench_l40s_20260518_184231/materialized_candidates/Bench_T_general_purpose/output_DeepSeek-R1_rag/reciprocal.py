import triton
import triton.language as tl
import torch

@triton.jit
def reciprocal_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DO_CAST: tl.constexpr,  # Whether to cast integral inputs to float32
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    if DO_CAST:
        input_values = input_values.to(tl.float32)
    output_values = 1.0 / input_values
    tl.store(output_ptr + offsets, output_values, mask=mask)

def reciprocal(input, *, out=None):
    # Determine output dtype based on input type
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        output_dtype = torch.get_default_dtype()
        do_cast = True
    else:
        output_dtype = input.dtype
        do_cast = False
    
    # Validate or create output tensor
    if out is not None:
        if out.dtype != output_dtype:
            raise TypeError(f"Expected out tensor to have dtype {output_dtype}, but got {out.dtype}")
        if out.shape != input.shape:
            raise RuntimeError(f"Shape mismatch: expected {input.shape}, but got {out.shape}")
        output = out
    else:
        output = torch.empty_like(input, dtype=output_dtype)
    
    n_elements = input.numel()
    if n_elements == 0:
        return output  # Handle empty tensor
    
    # Heuristic for block size; can be adjusted based on hardware specifics
    block_size = 1024
    grid_size = triton.cdiv(n_elements, block_size)
    
    reciprocal_kernel[grid_size](
        input, output, n_elements,
        BLOCK_SIZE=block_size,
        DO_CAST=do_cast
    )
    
    return output

if __name__ == "__main__":
    # Test cases
    # Integral input
    a_int = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    out_int = reciprocal(a_int)
    print(out_int.dtype)  # Should be torch.float32
    
    # Float input
    a_float = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out_float = reciprocal(a_float)
    print(out_float.dtype)  # Should be torch.float32
    
    # Using out parameter
    out = torch.empty_like(a_float)
    reciprocal(a_float, out=out)
    print(out)
