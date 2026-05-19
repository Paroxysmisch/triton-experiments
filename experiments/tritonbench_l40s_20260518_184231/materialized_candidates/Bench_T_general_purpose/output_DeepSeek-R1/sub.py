import torch
import triton
import triton.language as tl
from typing import Union, Optional, Number

def torch_dtype_to_tl(dtype):
    if dtype == torch.int8:
        return tl.int8
    elif dtype == torch.int16:
        return tl.int16
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.int64:
        return tl.int64
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float64:
        return tl.float64
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.complex64:
        return tl.complex64
    elif dtype == torch.complex128:
        return tl.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

@triton.jit
def sub_kernel(
    input_ptr,
    tmp_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    INPUT_TYPE: tl.constexpr,
    TMP_TYPE: tl.constexpr,
    OUTPUT_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask, dtype=INPUT_TYPE)
    tmp = tl.load(tmp_ptr + offsets, mask=mask, dtype=TMP_TYPE)
    output = input - tmp
    tl.store(output_ptr + offsets, output, mask=mask, dtype=OUTPUT_TYPE)

def sub(input: torch.Tensor, other: Union[torch.Tensor, Number], *, alpha: Number = 1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Convert other to a tensor if it's a number
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=input.device)
    
    # Convert alpha to a tensor on the same device as other
    alpha_tensor = torch.tensor(alpha, device=other.device)
    
    # Compute tmp = alpha * other
    tmp = alpha_tensor * other
    
    # Compute broadcasted shape
    try:
        broadcast_shape = torch.broadcast_shapes(input.shape, tmp.shape)
    except RuntimeError as e:
        raise RuntimeError(f"Shapes cannot be broadcasted: input {input.shape}, tmp {tmp.shape}") from e
    
    # Expand and make contiguous
    input_expanded = input.expand(broadcast_shape).contiguous()
    tmp_expanded = tmp.expand(broadcast_shape).contiguous()
    
    # Determine output dtype
    output_dtype = torch.result_type(input_expanded, tmp_expanded)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(broadcast_shape, dtype=output_dtype, device=input.device)
    else:
        if out.shape != broadcast_shape:
            raise RuntimeError(f"out shape {out.shape} does not match {broadcast_shape}")
        if out.dtype != output_dtype:
            raise RuntimeError(f"out dtype {out.dtype} does not match {output_dtype}")
        if not out.is_contiguous():
            raise RuntimeError("out must be contiguous")
    
    # Check CUDA device
    if input_expanded.device.type != 'cuda' or tmp_expanded.device.type != 'cuda' or out.device.type != 'cuda':
        raise RuntimeError("Tensors must be on CUDA device")
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Convert dtypes to Triton types
    input_tl_dtype = torch_dtype_to_tl(input_expanded.dtype)
    tmp_tl_dtype = torch_dtype_to_tl(tmp_expanded.dtype)
    output_tl_dtype = torch_dtype_to_tl(output_dtype)
    
    # Launch kernel
    sub_kernel[grid](
        input_expanded.data_ptr(),
        tmp_expanded.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
        INPUT_TYPE=input_tl_dtype,
        TMP_TYPE=tmp_tl_dtype,
        OUTPUT_TYPE=output_tl_dtype,
    )
    
    return out
