import logging
import triton
import triton.language as tl
import torch

def get_wrapper_code(func_str, func_name, arg_names, use_triton_kernel, kernel_name=None, grid=None, stream=None):
    arg_str = ', '.join(arg_names)
    wrapper_code = f"""
@torch.library.impl(lib, "{func_name}")
def {func_name}({arg_str}):
    logging.debug("GEMS {func_name} is called")
    if isinstance({arg_names[0]}, torch.Tensor) and isinstance({arg_names[1]}, torch.Tensor):
        input_shape = {arg_names[0]}.shape
        other_shape = {arg_names[1]}.shape
        if input_shape == other_shape:
            return {func_name}_kernel({arg_names[0]},{arg_names[1]})
        else:
            raise RuntimeError(f"Shape must be the same but got {input_shape} and {other_shape}")
    else:
        raise RuntimeError("Both inputs must be tensors")
"""

    if use_triton_kernel:
        kernel_code = f"""
@triton.jit
def {kernel_name}(input, other):
    return input & other
"""
        wrapper_code = kernel_code + wrapper_code
    return wrapper_code

def get_typed_wrapper_code(func_str, func_name, arg_names, use_triton_kernel, kernel_name=None, grid=None, stream=None):
    arg_str = ', '.join(arg_names)
    wrapper_code = f"""
@torch.library.impl(lib, "{func_name}")
def {func_name}({arg_str}):
    logging.debug("GEMS {func_name} is called")
    if isinstance({arg_names[0]}, torch.Tensor) and isinstance({arg_names[1]}, torch.Tensor):
        input_shape = {arg_names[0]}.shape
        other_shape = {arg_names[1]}.shape
        if input_shape == other_shape:
            if {arg_names[0]}.dtype in [torch.bool]:
                return {func_name}_kernel({arg_names[0]},{arg_names[1]})
            else:
                raise RuntimeError("Only boolean tensors are supported")
        else:
            raise RuntimeError(f"Shape must be the same but got {input_shape} and {other_shape}")
    else:
        raise RuntimeError("Both inputs must be tensors")
"""

    if use_triton_kernel:
        kernel_code = f"""
@triton.jit
def {kernel_name}(input, other):
    return input & other
"""
        wrapper_code = kernel_code + wrapper_code
    return wrapper_code

func_str = "Computes the bitwise AND of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND."
func_name = "bitwise_and"
arg_names = ["input", "other"]
use_triton_kernel = True
kernel_name = "bitwise_and_kernel"
grid = None
stream = None
wrapper_code = get_wrapper_code(func_str, func_name, arg_names, use_triton_kernel, kernel_name, grid, stream)
