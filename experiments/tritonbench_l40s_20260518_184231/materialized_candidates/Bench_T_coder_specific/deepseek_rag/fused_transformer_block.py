rms_norm(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    square_output = row * row
    rms = tl.sqrt(tl.sum(square_output)/n_cols + eps)
    normalized_output = row / (rms + eps)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, normalized_output, mask=col_offsets < n_cols)

def square(input):
    output = torch.empty_like(input)
    input_ptr = triton.ffi.from_numpy(input.numpy())
    output_ptr = triton.ffi.from_numpy(output.numpy())
    n_rows, n_cols = input.shape
    square_kernel[n_rows, BLOCK_SIZE](output_ptr, input_ptr, n_cols, n_cols, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

def mean_of_squares(input, eps=1e-5):
    output = torch.empty_like(input)
    input_ptr = triton.ffi.from_numpy(input.numpy())
    output_ptr = triton.ffi.from_numpy(output.numpy())
    n_rows, n_cols = input.shape
    mean_of_squares_kernel[n_rows, BLOCK_SIZE](output_ptr, input_ptr, n_cols, n_cols, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output

def rms(input, eps=1e-5):
    output = torch.empty_like(input)
    input_ptr = triton.ffi.from_numpy(input.numpy())
    output_ptr = triton.ffi.from_numpy(output.numpy())
    n_rows, n_cols = input.shape
    rms_kernel[n_rows, BLOCK_SIZE](output_ptr, input_ptr, n_cols, n_cols, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output

def rms_norm(input, eps=1e-5):
    output = torch.empty_like(input)
    input_ptr = triton.ffi.from_numpy(input.numpy())
    output_ptr = triton.ffi.from_numpy(output.numpy())
    n_rows, n_cols = input.shape
    rms_norm[n_rows, BLOCK_SIZE](output_ptr, input_ptr, n_cols, n_cols, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output|

---

TrionUser: You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Performs a sequence of operations commonly used in transformer models, combining matrix multiplication, softmax, dropout, another matrix multiplication, layer normalization, and addition (residual connection).
Wrapper Entry Information: fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None) -> Tensor; input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any number of batch dimensions.; weight1 (Tensor): Weight matrix of shape (D_in, D_k).; weight2 (Tensor): Weight matrix of shape (D_k, D_out).; residual (Tensor): Residual tensor to be added before layer normalization, must be broadcastable to the shape of Z_4.; dropout_p (float, optional): Probability of an element to be zeroed in the dropout layer. Default: 0.1.; eps (float, optional): A value added to the denominator for numerical stability in layer normalization. Default: 1e-5.; out (Tensor, optional): Output tensor. Ignored if None. Default: None.
Math: Given an input tensor X, weight matrices W_1 and W_2, and a residual tensor R, the function computes:

    \[
    \begin{align*}
    Z_1 &= X W_1 \\
    Z_2 &= \text{softmax}(Z_1) \\
    Z_3 &= \text{dropout}(Z_2, p) \\
    Z_4 &= Z_3 W_2 \\
    Y &= \text{LayerNorm}(Z_4 + R, \gamma, \beta, \epsilon)
    \end{align*}
    \]

    where:
    - \text{softmax}(Z) is applied along the last dimension.
    - \text{dropout}(Z, p) randomly zeroes elements of Z with probability p.
    - \text{LayerNorm} applies layer normalization with learnable parameters \gamma and \beta, and epsilon \epsilon for numerical stability.
    - R is the residual tensor added to Z_4 before layer normalization.
other: - The dimensions of `input` and `weight1` must be compatible for matrix multiplication: the last dimension of `input` must match the first dimension of `weight1`.
    - The output of the first matrix multiplication has shape `(*, N, D_k)`.
    - The `softmax` is applied along the last dimension (`dim=-1`).
    - The `dropout` is applied during training. Set `training=False` to disable dropout during evaluation.
    - The `layer_norm` is applied over the last dimension of the input tensor.
    - The `residual` tensor must be broadcastable to the shape of `z4`.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to create a suite of kernel functions for matrix operations. There are four main kernels: `square_kernel` computes the element-wise square of the input matrix. `mean_of_squares_kernel` computes the mean of the squares of each row. `rms_kernel` computes the root mean square of each row. `rms_norm` performs RMS normalization on the input matrix. These kernels leverage triton's parallel execution across rows and efficient memory handling. Each kernel takes in pointers to input/output memory, strides, the number of columns, and a block size to parallelize over. Some kernels also take an epsilon value for numerical stability. import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit
