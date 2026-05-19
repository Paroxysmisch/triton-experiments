import logging
import triton
import triton.language as tl
from .activation import register_triton_kernel

@register_triton_kernel(
    "leaky_relu",
    ["<float32>"],
    "<float32>",
    {
        "configs": [
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_stages=1),
        ],
        "constants": {},
    },
)
@triton.jit
def leaky_relu(dt):
    dt = tl.where(dt >= 0, dt, 0.01 * dt)
    return dt

def get_kernel_name(func_name):
    kernel_name = func_name.replace("forward", "")
    return kernel_name
