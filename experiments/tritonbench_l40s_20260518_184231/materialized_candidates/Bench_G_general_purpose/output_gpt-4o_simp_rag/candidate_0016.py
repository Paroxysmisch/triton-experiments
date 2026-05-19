import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe_kernel(x_ptr: tl.pointer_type,
                          weight_ptr: tl.pointer_type,
                          output_ptr: tl.pointer_type,
                          x_row_stride: tl.uint32,
                          weight_col_stride: tl.uint32,
                          H: tl.uint32,
                          eps: tl.float32,
                          apply_rbe: tl.constexpr,
                          rotary_emb_ptr: tl.pointer_type,
                          BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Load the input row
    x_row_start_ptr = x_ptr + row_idx * x_row_stride
    x_offsets = tl.arange(0, BLOCK_SIZE)
    x_mask = x_offsets < H
    x_row = tl.load(x_row_start_ptr + x_offsets, mask=x_mask, other=0)
    
    # Load the weight column
    weight_col_start_ptr = weight_ptr + col_idx * weight_col_stride
    weight_offsets = tl.arange(0, BLOCK_SIZE)
    weight_mask = weight_offsets < H
    weight_col = tl.load(weight_col_start_ptr + weight_offsets, mask=weight_mask, other=0)
    
    # Apply rotary embeddings if specified
    if apply_rbe:
        rotary_emb = tl.load(rotary_emb_ptr + x_offsets, mask=x_mask, other=1)
        x_row = x_row * rotary_emb

    # Compute dot product
    dot_product = tl.sum(x_row * weight_col)

    # RMS normalization
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)
    normalized_dot_product = dot_product / rms

    # Store the result
    output_ptr = output_ptr + row_idx * weight_col_stride + col_idx
    tl.store(output_ptr, normalized_dot_product)

def rms_matmul_rbe_wrapper(x, weight, apply_rbe=False, rotary_emb=None):
    assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous() and weight.is_contiguous(), "Expected contiguous tensors"

    B, H = x.shape
    _, D = weight.shape

    BLOCK_SIZE = triton.next_power_of_2(H)
    output = torch.empty((B, D), device=x.device, dtype=x.dtype)

    grid = (B, D)
    eps = 1e-9

    if apply_rbe:
        assert rotary_emb is not None, "Rotary embeddings are enabled, but no embeddings provided."
        assert rotary_emb.is_cuda and rotary_emb.is_contiguous(), "Expected CUDA contiguous rotary embeddings"

    rms_matmul_rbe_kernel[grid](
        x, weight, output,
        x.stride(0), weight.stride(1), H, eps,
        apply_rbe, rotary_emb if apply_rbe else None,
        num_warps=16, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
