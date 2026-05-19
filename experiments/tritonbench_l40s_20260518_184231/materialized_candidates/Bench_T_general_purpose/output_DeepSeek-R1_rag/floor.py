import torch
import triton
import triton.language as tl
from typing import Optional

device = 'cuda:0'

@triton.jit
def floor_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.floor(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def floor(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check if input is integer type and handle accordingly
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        if out is not None:
            if out.shape != input.shape:
                raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
            if out.dtype != input.dtype:
                raise TypeError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")
            if not out.is_cuda:
                raise RuntimeError("out tensor must be on CUDA")
            out.copy_(input)
            return out
        else:
            return input.clone()
    else:
        # Ensure input is on CUDA
        if not input.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA")
        N = input.numel()
        if N == 0:
            if out is not None:
                return out
            else:
                return torch.empty_like(input)
        # Prepare output tensor
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
            if output.shape != input.shape:
                raise ValueError(f"out shape {output.shape} does not match input shape {input.shape}")
            if output.dtype != input.dtype:
                raise TypeError(f"out dtype {output.dtype} does not match input dtype {input.dtype}")
            if not output.is_cuda:
                raise RuntimeError("out tensor must be on CUDA")
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        floor_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
        return output
