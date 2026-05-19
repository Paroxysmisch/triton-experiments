import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    X, G, H0, O,  # Pointers to the input and output tensors
    T, B, H,  # Shape of the tensors
    stride_xb, stride_xh,  # Strides for input tensor X
    stride_gb, stride_gh,  # Strides for input tensor G
    stride_hb, stride_hh,  # Strides for initial state H0
    stride_ob, stride_oh,  # Strides for output tensor O
    BLOCK_SIZE: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    if b >= B or h >= H:
        return

    # Pointers for the initial state
    h0_ptr = H0 + b * stride_hb + h * stride_hh

    # Pointers for the input and output tensors
    x_ptr = X + b * stride_xb + h * stride_xh
    g_ptr = G + b * stride_gb + h * stride_gh
    o_ptr = O + b * stride_ob + h * stride_oh

    # Load the initial state
    h_t = tl.load(h0_ptr)

    for t in range(T):
        # Load the input and gate values
        x_t = tl.load(x_ptr + t * stride_xh)
        g_t = tl.load(g_ptr + t * stride_gh)

        # Compute the output
        o_t = g_t * h_t + x_t

        # Store the output
        tl.store(o_ptr + t * stride_oh, o_t)

        # Update the state
        h_t = o_t

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    dX, dG, O, dO,  # Pointers to the gradients and output tensors
    T, B, H,  # Shape of the tensors
    stride_xb, stride_xh,  # Strides for gradient tensor dX
    stride_gb, stride_gh,  # Strides for gradient tensor dG
    stride_ob, stride_oh,  # Strides for output tensor O
    stride_do, stride_doh,  # Strides for gradient tensor dO
    BLOCK_SIZE: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    if b >= B or h >= H:
        return

    # Pointers for the input and output tensors
    x_ptr = dX + b * stride_xb + h * stride_xh
    g_ptr = dG + b * stride_gb + h * stride_gh
    o_ptr = O + b * stride_ob + h * stride_oh
    do_ptr = dO + b * stride_do + h * stride_doh

    # Initialize the gradients
    dX_t = 0.0
    dG_t = 0.0
    dO_t = 0.0

    for t in range(T - 1, -1, -1):
        # Load the output and gradient values
        o_t = tl.load(o_ptr + t * stride_oh)
        do_t = tl.load(do_ptr + t * stride_doh)

        # Compute the gradients
        dX_t = do_t
        dG_t = do_t * o_t

        # Store the gradients
        tl.store(x_ptr + t * stride_xh, dX_t)
        tl.store(g_ptr + t * stride_gh, dG_t)

import torch
from torch.autograd import Function

class FusedRecurrentHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None):
        B, T, H = x.shape
        device = x.device

        if initial_state is None:
            initial_state = torch.zeros((B, H), device=device)

        o = torch.empty((B, T, H), device=device)

        # Launch the Triton kernel
        grid = (B, H)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, initial_state, o,
            T, B, H,
            x.stride(0), x.stride(1),
            g.stride(0), g.stride(1),
            initial_state.stride(0), initial_state.stride(1),
            o.stride(0), o.stride(1),
            BLOCK_SIZE=1
        )

        ctx.save_for_backward(x, g, o)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        x, g, o = ctx.saved_tensors
        B, T, H = x.shape

        grad_x = torch.zeros_like(x)
        grad_g = torch.zeros_like(g)

        # Launch the Triton kernel
        grid = (B, H)
        fused_recurrent_hgrn_bwd_kernel[grid](
            grad_x, grad_g, o, grad_output,
            T, B, H,
            grad_x.stride(0), grad_x.stride(1),
            grad_g.stride(0), grad_g.stride(1),
            o.stride(0), o.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            BLOCK_SIZE=1
        )

        return grad_x, grad_g, None

def fused_recurrent_hgrn(x, g, initial_state=None, return_final_state=False):
    o = FusedRecurrentHGRNFunction.apply(x, g, initial_state)
    if return_final_state:
        final_state = o[:, -1, :]
        return o, final_state
    return o

import torch

# Example input tensors
B, T, H = 2, 10, 5
x = torch.randn((B, T, H), requires_grad=True)
g = torch.randn((B, T, H), requires_grad=True)

# Optional initial state
initial_state = torch.randn((B, H))

# Forward pass
o, final_state = fused_recurrent_hgrn(x, g, initial_state, return_final_state=True)

# Backward pass
loss = o.sum()
loss.backward()

print("Output:", o)
print("Final State:", final_state)
print("Gradient of x:", x.grad)
print("Gradient of g:", g.grad)
