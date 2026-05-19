import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(
    # Pointers to matrices
    x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, out_ptr,
    # Matrix dimensions
    batch_size, seq_len, state_size,
    # Strides for tensors
    stride_xb, stride_xs, stride_xh,
    stride_ob, stride_os, stride_oh,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load input x for current position
    x_offs = batch_idx * stride_xb + seq_idx * stride_xs
    x = tl.load(x_ptr + x_offs)
    
    # Initialize state vector h
    h = tl.zeros([state_size], dtype=tl.float32)
    
    # Load diagonal elements of A, B, C
    A = tl.load(A_ptr + tl.arange(0, state_size))
    B = tl.load(B_ptr + tl.arange(0, state_size))
    C = tl.load(C_ptr + tl.arange(0, state_size))
    delta = tl.load(delta_ptr + seq_idx)
    
    # Update state: h = A * h + B * x
    h = A * h + B * x
    
    # Compute output: y = C * h
    y = tl.sum(C * h)
    
    # Store result
    out_offs = batch_idx * stride_ob + seq_idx * stride_os
    tl.store(out_ptr + out_offs, y)

@triton.jit
def diag_ssm_forward_kernel_complex(
    # Pointers to matrices
    x_ptr, delta_ptr, A_real_ptr, A_imag_ptr, B_ptr, C_ptr, out_ptr,
    # Matrix dimensions
    batch_size, seq_len, state_size,
    # Strides for tensors
    stride_xb, stride_xs, stride_xh,
    stride_ob, stride_os, stride_oh,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load input
    x_offs = batch_idx * stride_xb + seq_idx * stride_xs
    x = tl.load(x_ptr + x_offs)
    
    # Initialize complex state vector
    h_real = tl.zeros([state_size], dtype=tl.float32)
    h_imag = tl.zeros([state_size], dtype=tl.float32)
    
    # Load matrices
    A_real = tl.load(A_real_ptr + tl.arange(0, state_size))
    A_imag = tl.load(A_imag_ptr + tl.arange(0, state_size))
    B = tl.load(B_ptr + tl.arange(0, state_size))
    C = tl.load(C_ptr + tl.arange(0, state_size))
    delta = tl.load(delta_ptr + seq_idx)
    
    # Complex multiplication for state update
    h_real_new = A_real * h_real - A_imag * h_imag + B * x
    h_imag_new = A_real * h_imag + A_imag * h_real
    
    h_real = h_real_new
    h_imag = h_imag_new
    
    # Compute output (real part only as per SSM convention)
    y = tl.sum(C * h_real)
    
    # Store result
    out_offs = batch_idx * stride_ob + seq_idx * stride_os
    tl.store(out_ptr + out_offs, y)

@triton.jit
def diag_ssm_backward_kernel(
    # Pointers to matrices
    grad_out_ptr, x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr,
    grad_x_ptr, grad_A_ptr, grad_B_ptr, grad_C_ptr,
    # Matrix dimensions
    batch_size, seq_len, state_size,
    # Strides
    stride_gb, stride_gs, stride_gh,
    stride_xb, stride_xs, stride_xh,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = seq_len - 1 - (pid % seq_len)  # Backward pass
    
    # Initialize gradients
    grad_h = tl.zeros([state_size], dtype=tl.float32)
    
    # Load matrices
    A = tl.load(A_ptr + tl.arange(0, state_size))
    B = tl.load(B_ptr + tl.arange(0, state_size))
    C = tl.load(C_ptr + tl.arange(0, state_size))
    
    # Load gradient of output
    grad_out_offs = batch_idx * stride_gb + seq_idx * stride_gs
    grad_out = tl.load(grad_out_ptr + grad_out_offs)
    
    # Load input
    x_offs = batch_idx * stride_xb + seq_idx * stride_xs
    x = tl.load(x_ptr + x_offs)
    
    # Backward pass computations
    grad_h = grad_h * A + grad_out * C
    
    # Compute gradients
    grad_A = grad_h * h
    grad_B = grad_h * x
    grad_C = grad_out * h
    
    # Store gradients
    tl.atomic_add(grad_A_ptr + tl.arange(0, state_size), grad_A)
    tl.atomic_add(grad_B_ptr + tl.arange(0, state_size), grad_B)
    tl.atomic_add(grad_C_ptr + tl.arange(0, state_size), grad_C)
    
    # Compute and store input gradient
    grad_x = tl.sum(grad_h * B)
    tl.store(grad_x_ptr + x_offs, grad_x)

class SSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, A, B, C):
        batch_size, seq_len = x.shape
        state_size = A.shape[0]
        is_complex = A.is_complex()
        
        # Allocate output tensor
        output = torch.empty_like(x)
        
        # Grid and block sizes
        grid = (batch_size * seq_len,)
        
        if not is_complex:
            diag_ssm_forward_kernel[grid](
                x, delta, A, B, C, output,
                batch_size, seq_len, state_size,
                x.stride(0), x.stride(1), 1,
                output.stride(0), output.stride(1), 1,
                BLOCK_SIZE=32
            )
        else:
            diag_ssm_forward_kernel_complex[grid](
                x, delta, A.real, A.imag, B, C, output,
                batch_size, seq_len, state_size,
                x.stride(0), x.stride(1), 1,
                output.stride(0), output.stride(1), 1,
                BLOCK_SIZE=32
            )
        
        ctx.save_for_backward(x, delta, A, B, C)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, delta, A, B, C = ctx.saved_tensors
        batch_size, seq_len = x.shape
        state_size = A.shape[0]
        
        # Initialize gradient tensors
        grad_x = torch.empty_like(x)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        grad_C = torch.zeros_like(C)
        
        # Grid for backward pass
        grid = (batch_size * seq_len,)
        
        if not A.is_complex():
            diag_ssm_backward_kernel[grid](
                grad_output, x, delta, A, B, C,
                grad_x, grad_A, grad_B, grad_C,
                batch_size, seq_len, state_size,
                grad_output.stride(0), grad_output.stride(1), 1,
                x.stride(0), x.stride(1), 1,
                BLOCK_SIZE=32
            )
        else:
            # Implement complex backward pass if needed
            pass
        
        return grad_x, None, grad_A, grad_B, grad_C

def ssm_forward(x, delta, A, B, C):
    return SSMFunction.apply(x, delta, A, B, C)
