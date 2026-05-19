import torch
import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(in_ptr, out_ptr, n_elements, d_model,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_tensor = tl.load(in_ptr + offsets, mask=mask).to(tl.float32)
    
    max_value = tl.max(input_tensor, axis=0)
    numerator = tl.exp(input_tensor - max_value)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    other = tl.load(in_ptr + offsets, mask=mask).to(tl.float32)
    mul_output = softmax_output * other
    
    tl.store(out_ptr + offsets, mul_output, mask=mask)
    
    
def softmax_mul(input, other, dim, dtype=None, out=None) -> torch.Tensor:
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    assert isinstance(other, torch.Tensor) or isinstance(
        other, (float, int)
    ), "The other must be a number or a tensor"

    if dtype is None:
        dtype = input.dtype

    assert dtype == input.dtype, "Only support casting to the same dtype now"
    assert other.device == input.device, "Input and other must on the same device"
    
    if other.device.type != "cuda":
        raise NotImplementedError("Only support cuda now")

    input = torch.squeeze(input, dim)
    if isinstance(other, torch.Tensor):
        other = torch.squeeze(other, dim)
    else:
        other = torch.tensor(other, device=input.device, dtype=dtype).squeeze(dim)

    assert other.shape == input.shape, "The shape of other and input must the same"

    n_elements = input.numel()
    out = torch.empty_like(input, dtype=dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    softmax_mul_kernel[grid](input, out, n_elements, input.shape[-1])
    
    return out
