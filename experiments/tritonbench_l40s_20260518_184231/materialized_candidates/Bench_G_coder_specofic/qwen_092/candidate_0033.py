import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_DMODEL': 64, 'BLOCK_HEAD': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_DMODEL': 128, 'BLOCK_HEAD': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_DMODEL': 256, 'BLOCK_HEAD': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_DMODEL': 512, 'BLOCK_HEAD': 1}, num_stages=1, num_warps=4),
    ],
    key=['seq_len', 'model_dim', 'head_num']
)
def destindex_copy_quantize_kv(
    K, 
    DestLoc, 
    Out, 
    Out_scale, 
    head_num, 
    seq_len, 
    model_dim
):
    """
    Wrapper function to invoke the Triton kernel `_fwd_kernel_destindex_copy_quantize_kv`.
    """
    # Define block and grid sizes
    BLOCK_DMODEL = 64
    BLOCK_HEAD = 1
    grid = (head_num, seq_len)
    block = (BLOCK_DMODEL * BLOCK_HEAD, 1, 1)
    
    # Invoke the kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid, block](
        K, 
        DestLoc, 
        Out, 
        Out_scale, 
        head_num, 
        seq_len, 
        model_dim, 
        BLOCK_DMODEL, 
        BLOCK_HEAD
    )

# Example usage
if __name__ == "__main__":
    # Example tensors
    K = torch.randn(1, 1024, 128, device='cuda')
    DestLoc = torch.randint(0, 1024, (1, 1024, 128), device='cuda')
    Out = torch.zeros(1024, 128, device='cuda', dtype=torch.int8)
    Out_scale = torch.zeros(1, 1024, 128, device='cuda')
    
    # Call the wrapper function
    destindex_copy_quantize_kv(
        K, 
        DestLoc, 
        Out, 
        Out_scale, 
        head_num=1, 
        seq_len=1024, 
        model_dim=128
    )
    
    # Print the results
    print("Quantized Output:", Out)
    print("Scale Factors:", Out_scale)
