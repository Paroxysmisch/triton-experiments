class FifthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z):
        # Allocate memory for outputs
        output = torch.zeros_like(x)
        d_output = torch.zeros_like(x)

        # Save context for backpropagation
        ctx.save_for_backward(x, y, z)

        # Call Triton kernels
        triton.launch(fifth_order_fwd, (output,), (x, y, z,), x.numel())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        x, y, z = ctx.saved_tensors

        # Allocate memory for gradients
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_z = torch.zeros_like(z)

        # Call Triton kernels
        triton.launch(fifth_order_bwd, (grad_x, grad_y, grad_z,), (x, y, z, grad_output,), x.numel())

        return grad_x, grad_y, grad_z
