import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Logics_ptr, V_ptr, Out_ptr, B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
                batch_stride, head_stride, seqlen, dim, BLOCK_SIZE: tl.constexpr):
    # Determine which batch and head this program is working on
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Compute offsets for the current batch and head
    batch_offset = batch_id * batch_stride
    head_offset = head_id * head_stride
    
    # Load auxiliary data
    B_Loc = tl.load(B_Loc_ptr + batch_id)
    B_Start_Loc = tl.load(B_Start_Loc_ptr + batch_id)
    B_Seqlen = tl.load(B_Seqlen_ptr + batch_id)
    
    # Initialize loop variables
    start_n = B_Start_Loc
    offs_n = tl.arange(0, BLOCK_SIZE)
    
    # Iterate over sequence length in blocks
    for _ in range(0, B_Seqlen, BLOCK_SIZE):
        # Compute the starting pointers for Logics and V
        Logics_ptrs = Logics_ptr + batch_offset + head_offset + (start_n + offs_n) * dim
        V_ptrs = V_ptr + batch_offset + head_offset + (start_n + offs_n) * dim
        
        # Mask for bounds checking
        mask = offs_n < seqlen
        
        # Load logits and compute max for numerical stability
        logits = tl.load(Logics_ptrs, mask=mask, other=-float('inf'))
        logits_minus_max = logits - tl.max(logits, axis=0)
        
        # Compute softmax normalization
        exp_logits = tl.exp(logits_minus_max)
        softmax_denominator = tl.sum(exp_logits, axis=0)
        softmax_values = exp_logits / softmax_denominator
        
        # Compute the weighted sum using V
        V_values = tl.load(V_ptrs, mask=mask)
        weighted_sum = tl.sum(softmax_values * V_values, axis=0)
        
        # Store the result in Out
        Out_ptrs = Out_ptr + batch_offset + head_offset + (start_n + offs_n) * dim
        tl.store(Out_ptrs, weighted_sum, mask=mask)
        
        # Move to the next block
        start_n += BLOCK_SIZE

def token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, batch_size, head_size, seqlen, dim):
    BLOCK_SIZE = 128  # Define block size for processing
    grid = (batch_size, head_size)  # 2D grid for batch and head dimensions
    
    # Execute the Triton kernel
    _fwd_kernel[grid](
        Logics,
        V,
        Out,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        Logics.stride(0),  # batch_stride
        Logics.stride(1),  # head_stride
        seqlen,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
torch.manual_seed(42)
batch_size = 8
head_size = 12
seqlen = 128
dim = 64

Logics = torch.randn(batch_size, head_size, seqlen, dim, device='cuda')
V = torch.randn(batch_size, head_size, seqlen, dim, device='cuda')
Out = torch.empty_like(Logics)

B_Loc = torch.randint(0, seqlen, (batch_size,), device='cuda')
B_Start_Loc = torch.randint(0, seqlen, (batch_size,), device='cuda')
B_Seqlen = torch.randint(1, seqlen, (batch_size,), device='cuda')

token_softmax_reducev_fwd(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, batch_size, head_size, seqlen, dim)
