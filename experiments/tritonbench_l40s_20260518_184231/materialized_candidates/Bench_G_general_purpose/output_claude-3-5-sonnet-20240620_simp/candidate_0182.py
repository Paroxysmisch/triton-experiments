import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    # Pointers to matrices
    x_ptr, g_ptr, h0_ptr, o_ptr, h_ptr,
    # Matrix dimensions
    batch_size, hidden_size, seq_len,
    # Strides for tensors
    stride_x_batch, stride_x_hidden, stride_x_seq,
    stride_g_batch, stride_g_hidden, stride_g_seq,
    stride_h_batch, stride_h_hidden,
    stride_o_batch, stride_o_hidden, stride_o_seq,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and hidden indices
    batch_idx = pid // (hidden_size // BLOCK_SIZE)
    hidden_offset = (pid % (hidden_size // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Bounds checking
    if batch_idx >= batch_size:
        return
        
    # Load initial state
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if h0_ptr:
        h_offs = batch_idx * stride_h_batch + hidden_offset
        mask = tl.arange(0, BLOCK_SIZE) < hidden_size
        h = tl.load(h0_ptr + h_offs, mask=mask)
    
    # Main loop over sequence length
    for t in range(seq_len):
        # Load inputs for current timestep
        x_offs = batch_idx * stride_x_batch + hidden_offset + t * stride_x_seq
        g_offs = batch_idx * stride_g_batch + hidden_offset + t * stride_g_seq
        
        mask = tl.arange(0, BLOCK_SIZE) < hidden_size
        x = tl.load(x_ptr + x_offs, mask=mask)
        g = tl.load(g_ptr + g_offs, mask=mask)
        
        # Compute next hidden state
        h = g * h + x
        
        # Store output and hidden state
        o_offs = batch_idx * stride_o_batch + hidden_offset + t * stride_o_seq
        tl.store(o_ptr + o_offs, h, mask=mask)
        
        if h_ptr:
            h_offs = batch_idx * stride_h_batch + hidden_offset
            tl.store(h_ptr + h_offs, h, mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    # Pointers to matrices
    grad_o_ptr, grad_x_ptr, grad_g_ptr,
    x_ptr, g_ptr, h0_ptr,
    # Matrix dimensions
    batch_size, hidden_size, seq_len,
    # Strides for tensors
    stride_grad_batch, stride_grad_hidden, stride_grad_seq,
    stride_x_batch, stride_x_hidden, stride_x_seq,
    stride_g_batch, stride_g_hidden, stride_g_seq,
    stride_h_batch, stride_h_hidden,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Compute batch and hidden indices
    batch_idx = pid // (hidden_size // BLOCK_SIZE)
    hidden_offset = (pid % (hidden_size // BLOCK_SIZE)) * BLOCK_SIZE
    
    if batch_idx >= batch_size:
        return
        
    # Initialize gradients
    grad_h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Backward pass through time
    for t in range(seq_len - 1, -1, -1):
        # Load gradients and states
        grad_offs = batch_idx * stride_grad_batch + hidden_offset + t * stride_grad_seq
        x_offs = batch_idx * stride_x_batch + hidden_offset + t * stride_x_seq
        g_offs = batch_idx * stride_g_batch + hidden_offset + t * stride_g_seq
        
        mask = tl.arange(0, BLOCK_SIZE) < hidden_size
        grad_o = tl.load(grad_o_ptr + grad_offs, mask=mask)
        x = tl.load(x_ptr + x_offs, mask=mask)
        g = tl.load(g_ptr + g_offs, mask=mask)
        
        # Add incoming gradient
        grad_h = grad_h + grad_o
        
        # Compute gradients
        grad_x = grad_h
        grad_g = grad_h * (x if t == 0 else h_prev)
        
        # Store gradients
        tl.store(grad_x_ptr + x_offs, grad_x, mask=mask)
        tl.store(grad_g_ptr + g_offs, grad_g, mask=mask)
        
        # Update for next iteration
        if t > 0:
            h_prev = x
            grad_h = grad_h * g

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None):
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Allocate output tensor
        output = torch.empty_like(x)
        final_state = torch.empty((batch_size, hidden_size), device=device)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(batch_size * hidden_size, meta['BLOCK_SIZE']),)
        
        fused_recurrent_hgrn_fwd_kernel[grid](
            x.contiguous(), g.contiguous(),
            initial_state.contiguous() if initial_state is not None else None,
            output, final_state,
            batch_size, hidden_size, seq_len,
            x.stride(0), x.stride(2), x.stride(1),
            g.stride(0), g.stride(2), g.stride(1),
            final_state.stride(0), final_state.stride(1),
            output.stride(0), output.stride(2), output.stride(1),
            BLOCK_SIZE=128,
        )
        
        # Save for backward
        ctx.save_for_backward(x, g, initial_state)
        return output, final_state
    
    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        x, g, initial_state = ctx.saved_tensors
        batch_size, seq_len, hidden_size = x.shape
        
        # Allocate gradient tensors
        grad_x = torch.empty_like(x)
        grad_g = torch.empty_like(g)
        
        # Launch backward kernel
        grid = lambda meta: (triton.cdiv(batch_size * hidden_size, meta['BLOCK_SIZE']),)
        
        fused_recurrent_hgrn_bwd_kernel[grid](
            grad_output.contiguous(), grad_x, grad_g,
            x.contiguous(), g.contiguous(),
            initial_state.contiguous() if initial_state is not None else None,
            batch_size, hidden_size, seq_len,
            grad_output.stride(0), grad_output.stride(2), grad_output.stride(1),
            x.stride(0), x.stride(2), x.stride(1),
            g.stride(0), g.stride(2), g.stride(1),
            grad_final_state.stride(0), grad_final_state.stride(1),
            BLOCK_SIZE=128,
        )
        
        return grad_x, grad_g, None

def fused_recurrent_hgrn(x, g, initial_state=None, return_state=False):
    """
    Fused Recurrent Hierarchical Gated Recurrent Network
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        g: Gate tensor of shape (batch_size, seq_len, hidden_size)
        initial_state: Optional initial hidden state of shape (batch_size, hidden_size)
        return_state: If True, returns final state along with output
        
    Returns:
        output: Output tensor of shape (batch_size, seq_len, hidden_size)
        final_state: (Optional) Final hidden state
    """
    output, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state)
    return (output, final_state) if return_state else output
