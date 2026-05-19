import torch
import triton
import triton.language as tl

# Triton kernel to perform the subtraction operation
@triton.jit
def sub_kernel(
    input_ptr, other_ptr, output_ptr,
    input_batch_stride, input_feature_stride,
    other_batch_stride, other_feature_stride,
    output_batch_stride, output_feature_stride,
    n_batches, n_features,
    alpha: tl.constexpr
):
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    input_row_offset = batch_idx * input_batch_stride
    other_row_offset = batch_idx * other_batch_stride
    output_row_offset = batch_idx * output_batch_stride
    
    input_ptr += input_row_offset
    other_ptr += other_row_offset
    output_ptr += output_row_offset
    
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_features,),
        strides=(input_feature_stride,),
        offsets=(feature_idx,),
        block_shape=(1,),
        order=(1,)
    )
    other_block_ptr = tl.make_block_ptr(
        base=other_ptr,
        shape=(n_features,),
        strides=(other_feature_stride,),
        offsets=(feature_idx,),
        block_shape=(1,),
        order=(1,)
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_features,),
        strides=(output_feature_stride,),
        offsets=(feature_idx,),
        block_shape=(1,),
        order=(1,)
    )
    
    input = tl.load(input_block_ptr)
    other = tl.load(other_block_ptr)
    result = input - alpha * other
    
    tl.store(output_block_ptr, result)

# Wrapper function for Triton kernel
def triton_sub(input: torch.Tensor, other: torch.Tensor, *, alpha=1, out: torch.Tensor = None):
    assert input.is_contiguous() and (other.is_contiguous() or isinstance(other, (int, float, complex))), \
        "Precondition failed"
    
    n_batches, n_features = input.shape
    assert input.shape == other.shape and input.shape == out.shape, \
        "Shape mismatch"
    
    if out is None:
        out = torch.empty_like(input)
    
    grid = lambda meta: (n_batches, triton.cdiv(n_features, meta["BLOCK_SIZE"]))
    sub_kernel[grid](
        input, other, out,
        input.stride(0), input.stride(1),
        other.stride(0), other.stride(1),
        out.stride(0), out.stride(1),
        n_batches, n_features,
        alpha=alpha
    )
    
    return out
