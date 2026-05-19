import triton
import triton.language as tl
import torch
from typing import Optional, Union, List

@triton.jit
def _fused_layer_norm_relu_linear_kernel(
    input_ptr,
