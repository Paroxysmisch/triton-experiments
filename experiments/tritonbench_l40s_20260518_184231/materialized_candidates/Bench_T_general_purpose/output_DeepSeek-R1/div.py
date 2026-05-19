import torch
import triton
import triton.language as tl

@triton.jit
def div_kernel(
    input_ptr, other_ptr, output_ptr,
    input_row_stride, other_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_row_stride  # Assuming 1D for simplicity

    input = tl.load(input_ptr + offsets * input_row_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_row_stride, mask=mask)
    result = input / other
    tl.store(output_ptr + offsets * output_row_stride, result, mask=mask)

@triton.jit
def div_rounding_kernel(
    input_ptr, other_ptr, output_ptr,
    rounding_mode,
    input_row_stride, other_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_row_stride

    input = tl.load(input_ptr + offsets * input_row_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_row_stride, mask=mask)
    result = input / other

    if rounding_mode == 'floor':
        result = tl.floor(result)
    elif rounding_mode == 'trunc':
        result = result - tl.trunc(result % 1.0)

    tl.store(output_ptr + offsets * output_row_stride, result, mask=mask)

@triton.jit
def div_complex_kernel(
    real_input_ptr, imag_input_ptr, real_other_ptr, imag_other_ptr,
    real_output_ptr, imag_output_ptr,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_stride

    a_real = tl.load(real_input_ptr + offsets, mask=mask)
    a_imag = tl.load(imag_input_ptr + offsets, mask=mask)
    b_real = tl.load(real_other_ptr + offsets, mask=mask)
    b_imag = tl.load(imag_other_ptr + offsets, mask=mask)

    denominator = b_real * b_real + b_imag * b_imag
    real = (a_real * b_real + a_imag * b_imag) / denominator
    imag = (a_imag * b_real - a_real * b_imag) / denominator

    tl.store(real_output_ptr + offsets, real, mask=mask)
    tl.store(imag_output_ptr + offsets, imag, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Convert other to tensor if necessary
    other = torch.as_tensor(other, dtype=input.dtype, device=input.device)

    # Broadcast input and other
    input_bc, other_bc = torch.broadcast_tensors(input, other)

    # Check for complex inputs and rounding_mode
    if rounding_mode is not None and (input_bc.is_complex() or other_bc.is_complex()):
        raise RuntimeError("div: rounding_mode is not supported for complex inputs")

    # Determine output dtype
    if rounding_mode is not None:
        if input_bc.dtype.is_complex or other_bc.dtype.is_complex:
            output_dtype = torch.result_type(input_bc, other_bc)
        else:
            promoted_type = torch.result_type(input_bc, other_bc)
            if promoted_type.is_floating_point:
                output_dtype = promoted_type
            else:
                output_dtype = torch.get_default_dtype()
    else:
        if input_bc.dtype.is_complex or other_bc.dtype.is_complex:
            output_dtype = torch.result_type(input_bc, other_bc)
        else:
            if input_bc.dtype.is_floating_point or other_bc.dtype.is_floating_point:
                output_dtype = torch.result_type(input_bc, other_bc)
            else:
                output_dtype = torch.get_default_dtype()

    # Create output tensor
    if out is not None:
        if out.dtype != output_dtype:
            raise RuntimeError("div: out tensor dtype does not match expected dtype")
        if out.shape != input_bc.shape:
            raise RuntimeError("div: out tensor shape does not match broadcasted shape")
        output = out
    else:
        output = torch.empty(input_bc.shape, dtype=output_dtype, device=input_bc.device)

    # Handle complex numbers
    if output.is_complex():
        real_input = input_bc.real.resolve_conj().float()
        imag_input = input_bc.imag.resolve_conj().float()
        real_other = other_bc.real.resolve_conj().float()
        imag_other = other_bc.imag.resolve_conj().float()
        real_output = output.real
        imag_output = output.imag

        assert real_input.is_contiguous() and imag_input.is_contiguous()
        assert real_other.is_contiguous() and imag_other.is_contiguous()
        assert real_output.is_contiguous() and imag_output.is_contiguous()

        n_elements = real_output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        div_complex_kernel[grid](
            real_input, imag_input, real_other, imag_other,
            real_output, imag_output,
            n_elements,
            BLOCK_SIZE=256,
        )
    else:
        # Convert to appropriate dtype
        input_flat = input_bc.to(output_dtype).view(-1)
        other_flat = other_bc.to(output_dtype).view(-1)
        output_flat = output.view(-1)

        if rounding_mode is None:
            kernel = div_kernel
            args = (input_flat, other_flat, output_flat,
                    input_flat.stride(0), other_flat.stride(0), output_flat.stride(0))
        else:
            if rounding_mode not in ['floor', 'trunc']:
                raise ValueError(f"div: unsupported rounding_mode {rounding_mode}")
            kernel = div_rounding_kernel
            args = (input_flat, other_flat, output_flat, rounding_mode,
                    input_flat.stride(0), other_flat.stride(0), output_flat.stride(0))

        n_elements = output_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        kernel[grid](*args, BLOCK_SIZE=256)

    return output
