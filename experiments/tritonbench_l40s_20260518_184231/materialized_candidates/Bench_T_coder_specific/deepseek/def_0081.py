import triton
from triton import Tensor
import triton.language as tl

@triton.jit
def _sigmoid_kernel(input_ptr, output_ptr, n):
    for i in range(n):
        x = input_ptr[i]
        output_ptr[i] = 1 / (1 + tl.exp(-x))

def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]):
    output = Tensor(input.shape, dtype=input.dtype)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    _sigmoid_kernel[output.numel](input, output, output_size)
    return output
