import triton
import triton.language as tl

# Define constants for fifth-order spherical harmonics
Y5_0 = 0.23873241463784303
Y5_1 = 0.3321295295080586
Y5_2 = 0.2728658343501178
Y5_3 = 0.16629392246050907
Y5_4 = 0.07291162439663284
Y5_5 = 0.02223924213364489

@triton.jit
def fifth_order_fwd(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    r2 = x * x + y * y + z * z
    r = tl.sqrt(r2)
    r_inv = 1.0 / r
    x_inv = x * r_inv
    y_inv = y * r_inv
    z_inv = z * r_inv

    # Compute spherical harmonics
    Y5_0_val = Y5_0 * (5 * z_inv * z_inv - 3) * (5 * z_inv * z_inv - 1)
    Y5_1_val = Y5_1 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv - y_inv)
    Y5_2_val = Y5_2 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv + x_inv)
    Y5_3_val = Y5_3 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv)
    Y5_4_val = Y5_4 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv)
    Y5_5_val = Y5_5 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv - y_inv)

    # Store results
    tl.store(output_ptr + offsets * 6 + 0, Y5_0_val, mask=mask)
    tl.store(output_ptr + offsets * 6 + 1, Y5_1_val, mask=mask)
    tl.store(output_ptr + offsets * 6 + 2, Y5_2_val, mask=mask)
    tl.store(output_ptr + offsets * 6 + 3, Y5_3_val, mask=mask)
    tl.store(output_ptr + offsets * 6 + 4, Y5_4_val, mask=mask)
    tl.store(output_ptr + offsets * 6 + 5, Y5_5_val, mask=mask)

@triton.jit
def fifth_order_bwd(
    grad_output_ptr, x_ptr, y_ptr, z_ptr, grad_x_ptr, grad_y_ptr, grad_z_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr + offsets * 6, mask=mask)

    r2 = x * x + y * y + z * z
    r = tl.sqrt(r2)
    r_inv = 1.0 / r
    x_inv = x * r_inv
    y_inv = y * r_inv
    z_inv = z * r_inv

    # Compute gradients
    grad_x = (
        Y5_1 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv) +
        Y5_2 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv) +
        Y5_3 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv) +
        Y5_4 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv) +
        Y5_5 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv)
    )
    grad_y = (
        Y5_1 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv - y_inv) +
        Y5_2 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv + x_inv) +
        Y5_3 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv) +
        Y5_4 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv) +
        Y5_5 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv)
    )
    grad_z = (
        Y5_0 * (5 * z_inv * z_inv - 3) * (5 * z_inv * z_inv - 1) +
        Y5_1 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv - y_inv) +
        Y5_2 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv + x_inv) +
        Y5_3 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv + y_inv) +
        Y5_4 * (5 * z_inv * z_inv - 1) * (3 * z_inv * y_inv - x_inv) +
        Y5_5 * (5 * z_inv * z_inv - 1) * (3 * z_inv * x_inv - y_inv)
    )

    # Store gradients
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_y_ptr + offsets, grad_y, mask=mask)
    tl.store(grad_z_ptr + offsets, grad_z, mask=mask)

import torch
from torch.autograd import Function

class FifthOrderSphericalHarmonic(Function):
    @staticmethod
    def forward(ctx, x, y, z):
        n_elements = x.numel()
        output = torch.empty((n_elements, 6), device=x.device, dtype=x.dtype)

        # Launch the Triton kernel
        grid = (n_elements + 1024 - 1) // 1024
        fifth_order_fwd[grid, 1024](
            x, y, z, output,
            n_elements, 1024
        )

        ctx.save_for_backward(x, y, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        n_elements = x.numel()
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_z = torch.zeros_like(z)

        # Launch the Triton kernel
        grid = (n_elements + 1024 - 1) // 1024
        fifth_order_bwd[grid, 1024](
            grad_output, x, y, z, grad_x, grad_y, grad_z,
            n_elements, 1024
        )

        return grad_x, grad_y, grad_z

# Example usage
if __name__ == "__main__":
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    z = torch.randn(1024, device='cuda')

    output = FifthOrderSphericalHarmonic.apply(x, y, z)
    print(output)

    # Compute gradients
    output.sum().backward()
    print(x.grad)
    print(y.grad)
    print(z.grad)
