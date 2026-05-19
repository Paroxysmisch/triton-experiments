import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                           batch_size, dim, BLOCK_SIZE: tl.constexpr):
    # Get block index and offsets
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    
    # Load initial state and Lambda
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)
    
    # Main loop over sequence length
    for t in range(length):
        offsets = t * batch_size * dim + col_offsets
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        s = s * Lambda + x  # Core SSM update
        tl.store(y_ptr + offsets, s, mask=mask)

class _ssm_forward(torch.autograd.Function):
    BLOCK_SIZE = 128  # Empirically good for RTX 3090
    
    @staticmethod
    def forward(ctx, s, x, Lambda):
        # Input validation
        assert s.is_contiguous() and x.is_contiguous() and Lambda.is_contiguous()
        
        # Setup dimensions
        length, batch_size, dim = x.shape
        n = batch_size * dim
        y = torch.zeros_like(x)
        
        # Configure grid
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        
        # Launch appropriate kernel based on dtype
        if Lambda.dtype == torch.complex64:
            diag_ssm_forward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(x),
                torch.view_as_real(y), torch.view_as_real(Lambda),
                length, batch_size, dim, _ssm_forward.BLOCK_SIZE)
        else:
            diag_ssm_forward_kernel[grid](
                s, x, Lambda, y, length, batch_size, dim, _ssm_forward.BLOCK_SIZE)
            
        ctx.save_for_backward(s, y, Lambda)
        return y
