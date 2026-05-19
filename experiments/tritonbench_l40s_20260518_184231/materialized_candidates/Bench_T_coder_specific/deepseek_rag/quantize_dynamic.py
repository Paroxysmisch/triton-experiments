import triton.language as tl

@triton.jit
def dynamic_quantize_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    scale_ptr,   # Pointer to scale tensor
    stride_outputm,  # Stride for output in dimension m
    stride_outputn,  # Stride for output in dimension n
    stride_inputm,   # Stride for input in dimension m
    stride_inputn,   # Stride for input in dimension n
    n_elements,  # Number of elements to process
    M: tl.constexpr,  # Number of rows (tokens)
    N: tl.constexpr,  # Number of columns (hidden size)
):
    # Dynamic quantization kernel
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, N)
    mask = offsets < n_elements
    input_ptrs = input_ptr + pid * stride_inputm + offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=1e-6)
    abs_max_f = tl.reduce(input_vals, 0, _abs_max)
    dynamic_per_token_scale = 127.0 / abs_max_f
    precison_mask = tl.where(input_vals > 0, 0.5, -0.5)
    output_vals = (input_vals * dynamic_per_token_scale + precison_mask).to(tl.int8)
    output_ptrs = output_ptr + pid * stride_outputm + offsets
    tl.store(output_ptrs, output_vals, mask=mask)
    tl.store(scale_ptr + pid, abs_max_f / 127.0)
