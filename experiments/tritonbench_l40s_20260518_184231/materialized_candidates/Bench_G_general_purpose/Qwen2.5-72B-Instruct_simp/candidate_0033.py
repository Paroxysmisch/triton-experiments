import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr,  # Pointer to the input tensor K
    DestLoc_ptr,  # Pointer to the destination indices
    Out_ptr,  # Pointer to the output tensor
    Out_scale_ptr,  # Pointer to the output scale
    seq_len,  # Sequence length
    block_size: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size

    # Compute the scale
    max_val = -float('inf')
    min_val = float('inf')
    for i in range(block_start, block_start + block_size):
        if i < seq_len:
            k_val = tl.load(K_ptr + i)
            max_val = tl.max(max_val, k_val)
            min_val = tl.min(min_val, k_val)

    scale = (max_val - min_val) / 255.0

    # Write the scale to the output scale tensor
    tl.store(Out_scale_ptr + pid, scale)

    # Perform the copy and quantization
    for i in range(block_start, block_start + block_size):
        if i < seq_len:
            k_val = tl.load(K_ptr + i)
            dest_index = tl.load(DestLoc_ptr + i)
            quantized_val = tl.round((k_val - min_val) / scale)
            quantized_val = tl.to_int8(quantized_val)
            tl.store(Out_ptr + dest_index, quantized_val)

import torch
import triton
import triton.language as tl

def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = K.shape[0]
    block_size = 128  # Adjust block size as needed

    # Launch the kernel
    grid = (seq_len + block_size - 1) // block_size
    _fwd_kernel_destindex_copy_quantize_kv[grid, block_size](
        K,  # Input tensor K
        DestLoc,  # Destination indices
        Out,  # Output tensor
        Out_scale,  # Output scale tensor
        seq_len,  # Sequence length
        block_size  # Block size
    )

# Example usage
if __name__ == "__main__":
    seq_len = 1024
    K = torch.randn(seq_len, dtype=torch.float32, device='cuda')
    DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32, device='cuda')
    Out = torch.zeros(seq_len, dtype=torch.int8, device='cuda')
    Out_scale = torch.zeros((seq_len + 127) // 128, dtype=torch.float32, device='cuda')

    destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale)

    print("Quantized Output:", Out)
    print("Scales:", Out_scale)
