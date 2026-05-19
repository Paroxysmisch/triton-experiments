import triton
import triton.language as tl
import torch

@triton.jit
def _sgmv_expand_slice_kernel(
    data_ptr, data_batch_stride, data_seq_stride,
    lora_indices_ptr, lora_indices_stride,
    lora_weights_ptr, lora_weights_stride,
    output_ptr, output_batch_stride, output_seq_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the start of the block in the output matrix
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Compute the pointers for the input data and output
    data_ptrs = data_ptr + start_m * data_seq_stride + start_n
    output_ptrs = output_ptr + start_m * output_seq_stride + start_n
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, BLOCK_K):
        # Load LoRA indices
        lora_index = tl.load(lora_indices_ptr + k * lora_indices_stride)
        
        # Load LoRA weights
        lora_weight = tl.load(lora_weights_ptr + lora_index * lora_weights_stride)
        
        # Load data
        data = tl.load(data_ptrs + k * data_seq_stride)
        
        # Perform the multiplication and accumulation
        acc += data * lora_weight
    
    # Store the result
    tl.store(output_ptrs, acc)

def _sgmv_expand_slice(
    data: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_weights: torch.Tensor,
    output: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128
):
    # Ensure data is contiguous
    data = data.contiguous()
    lora_indices = lora_indices.contiguous()
    lora_weights = lora_weights.contiguous()
    output = output.contiguous()
    
    # Get strides
    data_batch_stride, data_seq_stride = data.stride()
    lora_indices_stride = lora_indices.stride(0)
    lora_weights_stride = lora_weights.stride(0)
    output_batch_stride, output_seq_stride = output.stride()
    
    # Define grid size
    grid = (output.shape[0] // BLOCK_M, output.shape[1] // BLOCK_N)
    
    # Launch kernel
    _sgmv_expand_slice_kernel[grid](
        data.data_ptr(), data_batch_stride, data_seq_stride,
        lora_indices.data_ptr(), lora_indices_stride,
        lora_weights.data_ptr(), lora_weights_stride,
        output.data_ptr(), output_batch_stride, output_seq_stride,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

# Example usage
batch_size = 32
seq_len = 128
dim = 64

data = torch.randn((batch_size, seq_len, dim), dtype=torch.float32, device='cuda')
lora_indices = torch.randint(0, dim, (dim,), dtype=torch.int32, device='cuda')
lora_weights = torch.randn((dim,), dtype=torch.float32, device='cuda')
output = torch.zeros((batch_size, seq_len, dim), dtype=torch.float32, device='cuda')

_sgmv_expand_slice(data, lora_indices, lora_weights, output)
