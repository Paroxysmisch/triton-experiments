import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    dim,
    alpha,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_DIM: tl.constexpr
):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    if HAS_DIM:
        o_d = i_d * B + tl.arange(0, B)
        m_d = o_d < D
    else:
        o_d = tl.arange(0, D)
        m_d = True

    b_input = tl.load(input_ptr + i_n * D + o_d, mask=m_d, other=0)
    b_other = tl.load(other_ptr + o_d, mask=m_d, other=0)
    b_output = tl.sum(b_input + alpha * b_other, 0) / (D if HAS_DIM else D * B)
    tl.store(output_ptr + i_n * (D if HAS_DIM else 1), b_output)


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if out is not None:
        assert out.is_cuda and out.dtype == torch.float32, "out must be a CUDA tensor with float32 dtype"
        output = out
    else:
        output = input.new_empty(input.shape if dim is None else input.shape[:dim] + input.shape[dim+1:], dtype=torch.float32)

    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
        output = output.to(dtype)

    N, D = input.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    add_mean_kernel[(N, ND)](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=output.data_ptr(),
        dim=dim,
        alpha=alpha,
        D=D,
        B=B,
        HAS_DIM=dim is not None
    )

    if keepdim:
        output = output.unsqueeze(dim)

    return output
