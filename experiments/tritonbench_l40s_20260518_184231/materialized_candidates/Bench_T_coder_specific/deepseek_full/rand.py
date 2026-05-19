import torch
import triton
import triton.language as tl
from torch.testing import assert_close

def test_rand_wrapper(func_inputs, func_correct_outputs, func_extra_args, func_kwargs):
    # Extract arguments
    size = func_inputs[0]
    generator = func_kwargs.get("generator", None)
    out = func_kwargs.get("out", None)
    dtype = func_kwargs.get("dtype", None)
    layout = func_kwargs.get("layout", torch.strided)
    device = func_kwargs.get("device", None)
    requires_grad = func_kwargs.get("requires_grad", False)
    pin_memory = func_kwargs.get("pin_memory", False)

    # Call the function
    result = torch.rand(
        *size,
        generator=generator,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )

    # Check result
    assert_close(result, func_correct_outputs[0])

    # Generate Triton wrapper
    triton_wrapper = TritonWrapper()
    triton_wrapper.generate_wrapper(
        torch.rand,
        signature=[size],
        kwargs=func_kwargs,
        extra_args=func_extra_args,
    )

    # Verify if Triton wrapper aligns with the provided function inputs
    assert_close(
        triton_wrapper.run(size, **func_kwargs),
        func_correct_outputs[0],
    )
