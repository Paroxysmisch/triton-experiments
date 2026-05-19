where:

y_{i} = \frac{\exp(\log(x_{i}))}{\sum_j \exp(\log(x_{j}))} = \frac{x_i}{\sum_j x_j}
other: The function handles optional data type casting to prevent overflow and allows specifying the dimension for softmax application.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement a streaming logsumexp kernel that computes the log of the sum of exponentials of input elements along a specified axis. The kernel takes input tensor X, output tensor OUT, strides for input and output tensors, the size of the reduction axis N, and a block size BLOCK_N. The kernel iterates over blocks of the input, computes the maximum value for numerical stability, and accumulates the exponentials. The result is stored in the output tensor. A wrapper function logsumexp is provided to reshape the input and call the kernel. Additionally, a softmax function softmax_ is implemented using the logsumexp kernel to normalize input tensor x along the last dimension. import triton
import triton.language as tl
import torch

# Triton kernel for streaming logsumexp
@triton.jit
def _logsumexp(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        # correct the current sum and update the max
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    rm = tl.program_id(0) + tl.arange(0, 1)
    OUT = OUT + rm * out_stride
    tl.store(OUT, out)

# Function to call the Triton kernel
def logsumexp(input):
    assert input.is_cuda
    *dims, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(*dims).view(-1)
    M = input.shape[0]
    _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                     BLOCK_N=4096, num_warps=4)
    return out.view(*dims)

# Softmax function using the logsumexp Triton kernel
def softmax_(x):
    if not x.is_cuda:
        return torch.softmax(x, dim=-1, out=x)
    c = logsumexp(x)
    return x.sub_(c[..., None]).exp_()
</s>
<|user|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Applies the natural logarithm element-wise on the input tensor, followed by applying the softmax function along the specified dimension. This combined operation scales input values to a range between 0 and 1, summing to 1 after the logarithmic transformation. It allows transformation of the input tensor into a probability distribution.
Wrapper Entry Information: def softmax_log(input, dim=-1, dtype=None) -> Tensor: 
Args:
    input (Tensor): The input tensor on which logarithm and softmax are applied.
    dim (int): The dimension along which softmax will be computed. Default: -1.
    dtype (:class:`torch.dtype`, optional): The desired data type of the returned tensor. If specified, the input tensor is cast to :attr:`dtype` before the operation is performed. Useful for preventing data type overflows. Default: None.
Example:
    >>> import torch
    >>> import torch.nn.functional as F
    >>> # Define input tensor
    >>> input = torch.rand(3, 4) * 10
    >>> # Apply softmax_log along the last dimension
    >>> result = softmax_log(input, dim=1)
    >>> result
    tensor([[0.1829, 0.1782, 0.2783, 0.3606],
            [0.3119, 0.1724, 0.3256, 0.1900],
            [0.2057, 0.2166, 0.2991, 0.2786]])

    >>> # Apply softmax_log along a different dimension
    >>> result = softmax_log(input, dim=0)
    >>> result
    tensor([[0.3122, 0.4444, 0.2720, 0.2159],
            [0.3879, 0.2167, 0.4226, 0.2165],
            [0.2999, 0.3389, 0.3055, 0.5676]])
Math: out = Softmax(log(input))

where:

y_{i} = \frac{\exp(\log(x_{i}))}{\sum_j \exp(\log(x_{j}))} = \frac{x_i}{\sum_j x_j}
other: The function handles optional data type casting to prevent overflow and allows specifying the dimension for softmax application.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement a streaming logsumexp kernel that computes the log of the sum of exponentials of input elements along a specified axis. The kernel takes input tensor X, output tensor OUT, strides for input and output tensors, the size of the reduction axis N, and a block size BLOCK_N. The kernel iterates over blocks of the input, computes the maximum value for numerical stability, and accumulates the exponentials. The result is stored in the output tensor. A wrapper function logsumexp is provided to reshape the input and call the kernel. Additionally, a softmax function softmax_ is implemented using the logsumexp kernel to normalize input tensor x along the last dimension. import triton
import triton.language as tl
import torch

# Triton kernel for streaming logsumexp
@triton.jit
def _logsumexp(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        # correct the current sum and update the max
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(
