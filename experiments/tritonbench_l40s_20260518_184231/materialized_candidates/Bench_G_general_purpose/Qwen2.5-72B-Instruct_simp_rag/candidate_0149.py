import triton
import triton.language as tl
import torch

# Triton kernel to double the values in the input tensor and store the result in a new tensor
@triton.jit
def mul2_kernel(X, stride_xm, Z, stride_zn, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_ptrs = X + offsets * stride_xm
    z_ptrs = Z + offsets * stride_zn
    x = tl.load(x_ptrs, mask=mask)
    z = x * 2
    tl.store(z_ptrs, z, mask=mask)

# Triton kernel to double the values in the input tensor in-place
@triton.jit
def mul2_inplace_kernel(X, stride_xm, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_ptrs = X + offsets * stride_xm
    x = tl.load(x_ptrs, mask=mask)
    z = x * 2
    tl.store(x_ptrs, z, mask=mask)

# Wrapper function for triton_mul2
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    z = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 1024),)
    mul2_kernel[grid](x, x.stride(0), z, z.stride(0), n_elements, BLOCK_SIZE=1024)
    return z

# Wrapper function for triton_mul2_inplace
def triton_mul2_inplace(x: torch.Tensor) -> None:
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    mul2_inplace_kernel[grid](x, x.stride(0), n_elements, BLOCK_SIZE=1024)

# Example usage
if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4, 5], device='cuda')
    print("Original tensor:", x)

    # Using triton_mul2
    z = triton_mul2(x)
    print("Result of triton_mul2:", z)

    # Using triton_mul2_inplace
    triton_mul2_inplace(x)
    print("Result of triton_mul2_inplace:", x)
