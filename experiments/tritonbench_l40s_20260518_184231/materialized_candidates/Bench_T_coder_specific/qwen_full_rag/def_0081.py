import triton
import triton.language as tl
from .utils import get_tensor_ndim, promote_to_tensor
from ..functional import adaptive_avg_pool2d, sigmoid


@triton.jit
def _jit_sigmoid_adaptive_avg_pool2d(input, output_size):
    H, W = input.shape[-2:]
    h, w = divmod(output_size, H * W)[::-1]

    input = input.reshape((-1, h, H // h, w, W // w))
    input = input.sum((2, 4))
    input = input.to(tl.float32)
    input = sigmoid(input)
    return input


def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    assert get_tensor_ndim(input) == 4, f"only accepts 4D tensor now"
    output_size = promote_to_tensor(output_size, dtype=tensor_int_dtype)
    return _jit_sigmoid_adaptive_avg_pool2d(input, output_size)
