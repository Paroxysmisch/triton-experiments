import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(
    input_ptr, output_ptr, state_ptr, matrix_ptr,
    BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch index and sequence length
    batch_idx = pid // SEQ_LEN
    seq_idx = pid % SEQ_LEN
    
    # Compute pointer offsets
    input_offset = batch_idx * SEQ_LEN * HIDDEN_SIZE + seq_idx * HIDDEN_SIZE
    output_offset = input_offset
    state_offset = batch_idx * HIDDEN_SIZE
    
    # Load input and state
    input = tl.load(input_ptr + input_offset)
    state = tl.load(state_ptr + state_offset)
    
    # Load diagonal matrix
    diag_matrix = tl.load(matrix_ptr + seq_idx * HIDDEN_SIZE)
    
    # Compute new state
    new_state = state * diag_matrix + input
    
    # Store new state and output
    tl.store(state_ptr + state_offset, new_state)
    tl.store(output_ptr + output_offset, new_state)

@triton.jit
def diag_ssm_backward_kernel(
    grad_output_ptr, grad_input_ptr, grad_state_ptr, matrix_ptr,
    BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch index and sequence length
    batch_idx = pid // SEQ_LEN
    seq_idx = pid % SEQ_LEN
    
    # Compute pointer offsets
    grad_output_offset = batch_idx * SEQ_LEN * HIDDEN_SIZE + seq_idx * HIDDEN_SIZE
    grad_input_offset = grad_output_offset
    grad_state_offset = batch_idx * HIDDEN_SIZE
    
    # Load grad output and state
    grad_output = tl.load(grad_output_ptr + grad_output_offset)
    grad_state = tl.load(grad_state_ptr + grad_state_offset)
    
    # Load diagonal matrix
    diag_matrix = tl.load(matrix_ptr + seq_idx * HIDDEN_SIZE)
    
    # Compute grad input and new grad state
    grad_input = grad_output
    new_grad_state = grad_state * diag_matrix + grad_output
    
    # Store grad input and new grad state
    tl.store(grad_input_ptr + grad_input_offset, grad_input)
    tl.store(grad_state_ptr + grad_state_offset, new_grad_state)

# Complex versions of these kernels would involve handling real and imaginary parts separately.

import torch
from torch.autograd import Function

class _SSMForward(Function):
    @staticmethod
    def forward(ctx, input, state, matrix):
        # Allocate output tensor
        output = torch.empty_like(input)
        
        # Define grid and block sizes
        BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE = input.shape
        BLOCK_SIZE = 128  # Example block size
        
        # Launch Triton kernel
        diag_ssm_forward_kernel[(BATCH_SIZE * SEQ_LEN,)](
            input, output, state, matrix,
            BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save for backward
        ctx.save_for_backward(input, state, matrix)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, state, matrix = ctx.saved_tensors
        
        # Allocate gradient tensors
        grad_input = torch.empty_like(input)
        grad_state = torch.empty_like(state)
        
        # Define grid and block sizes
        BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE = grad_output.shape
        BLOCK_SIZE = 128  # Example block size
        
        # Launch Triton kernel for backward pass
        diag_ssm_backward_kernel[(BATCH_SIZE * SEQ_LEN,)](
            grad_output, grad_input, grad_state, matrix,
            BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_input, grad_state, None  # No gradient for matrix

# Example usage:
# input = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
# state = torch.zeros(batch_size, hidden_size, device='cuda')
# matrix = torch.randn(seq_len, hidden_size, device='cuda')
# output = _SSMForward.apply(input, state, matrix)
