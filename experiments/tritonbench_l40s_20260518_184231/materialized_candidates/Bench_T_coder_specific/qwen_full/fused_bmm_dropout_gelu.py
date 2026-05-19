import torch
import triton
import triton.language as tl
from triton.language.libdevice import gelu, gelu_approx_tanh

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, out_ptr, input1_batch_stride, input1_n_stride,
    input1_m_stride, input2_batch_stride, input2_m_stride, input2_p_stride,
    out_batch_stride, out_n_stride, out_p_stride, batch_size, n_size, m_size,
    p_size, dropout_prob, is_training, inplace, approximate_gelu: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr):
    pid = tl.program_id(axis=0)
    dropout_hash = tl.program_id(axis=1)
    random_hash = tl.program_id(axis=2)

    rand_offsets = random_hash * 0

    input1_batch_offset = (pid * BLOCK_SIZE_N * BLOCK_SIZE_M) // n_size // m_size
    input2_batch_offset = (pid * BLOCK_SIZE_N * BLOCK_SIZE_P) // n_size // p_size

    input1_batch_ptr = input1_ptr + input1_batch_offset * input1_batch_stride
    input2_batch_ptr = input2_ptr + input2_batch_offset * input2_batch_stride

    out_batch_ptr = out_ptr + input1_batch_offset * out_batch_stride

    input1_ptrs = input1_batch_ptr + (
        (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_M +
         tl.arange(0, BLOCK_SIZE_M)) // m_size // n_size)
    input2_ptrs = input2_batch_ptr + (
        (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_P +
         tl.arange(0, BLOCK_SIZE_P)) // p_size // n_size)

    out_ptrs = out_batch_ptr + (
        (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_P +
         tl.arange(0, BLOCK_SIZE_P)) // p_size // n_size)

    if inplace:
        input1_ptrs = input1_ptrs.to(tl.int64)
        out_ptrs = out_ptrs.to(tl.int64)
        input1_ptrs = input1_ptrs + (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_M +
                                    tl.arange(0, BLOCK_SIZE_M)) // m_size
        input1_ptrs = input1_ptrs.to(tl.pointer_type(tl.float32))
        input1_ptrs = tl.make_block_ptr(input1_ptrs, (n_size, m_size), (1, 0),
                                        (0, 0), (BLOCK_SIZE_N, BLOCK_SIZE_M),
                                        (1, 0))
        input2_ptrs = input2_ptrs + (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_P +
                                    tl.arange(0, BLOCK_SIZE_P)) // p_size
        input2_ptrs = input2_ptrs.to(tl.pointer_type(tl.float32))
        input2_ptrs = tl.make_block_ptr(input2_ptrs, (m_size, p_size), (0, 1),
                                        (0, 0), (BLOCK_SIZE_N, BLOCK_SIZE_P),
                                        (0, 1))
        out_ptrs = tl.make_block_ptr(out_ptrs, (n_size, p_size), (1, 0),
                                     (0, 0), (BLOCK_SIZE_N, BLOCK_SIZE_P),
                                     (1, 0))
    else:
        input1_ptrs = input1_ptrs + (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_M +
                                    tl.arange(0, BLOCK_SIZE_M)) // m_size
        input2_ptrs = input2_ptrs + (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_P +
                                    tl.arange(0, BLOCK_SIZE_P)) // p_size

    n_range = tl.arange(0, BLOCK_SIZE_N)
    m_range = tl.arange(0, BLOCK_SIZE_M)
    p_range = tl.arange(0, BLOCK_SIZE_P)

    input1 = tl.load(input1_ptrs,
                     mask=(n_range[:, None] < n_size) &
                     (m_range[None, :] < m_size),
                     other=0.0)
    input2 = tl.load(input2_ptrs,
                     mask=(m_range[:, None] < m_size) &
                     (p_range[None, :] < p_size),
                     other=0.0)

    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    matmul_result = tl.dot(input1, input2)

    if not inplace:
        tl.store(out_ptrs,
                 matmul_result,
                 mask=(n_range[:, None] < n_size) &
                 (p_range[None, :] < p_size))
        out_ptrs += (tl.arange(0, BLOCK_SIZE_N) * BLOCK_SIZE_P +
                     tl.arange(0, BLOCK_SIZE_P)) // p_size // n_size

    if is_training:
        dropout_mask = ((rand_ops.rand(
            rand_offsets,
            random_hash,
            BLOCK_SIZE_N * BLOCK_SIZE_P,
            approximate=approximate_gelu) > dropout_prob)
                        .to(tl.float32))
    else:
        dropout_mask = 1.0

    if approximate_gelu == 'tanh':
        matmul_result = gelu_approx_tanh(matmul_result)
    else:
        matmul_result = gelu(matmul_result)

    matmul_result *= dropout_mask

    if inplace:
        tl.store(input1_ptrs,
                 matmul_result,
                 mask=(n_range[:, None] < n_size) &
                 (m_range[None, :] < m_size))
    else:
        tl.store(out_ptrs,
                 matmul_result,
                 mask=(n_range[:, None] < n_size) &
                 (p_range[None, :] < p_size))


def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False,
                          approximate='none', *, out=None):
    check_bmm_compatible(input1, input2)

    input1 = input1.contiguous()
    input2 = input2.contiguous()

    if out is not None:
        out = out.contiguous()

    assert input1.shape[2] == input2.shape[1], "incompatible dimensions"
    assert input1.is_contiguous(), "matrix A must be contiguous"
    assert input2.is_contiguous(), "matrix B must be contiguous"

    batch_size, n_size, m_size = input1.shape
    m_size, p_size = input2.shape

    output = torch.empty((batch_size, n_size, p_size), dtype=input1.dtype,
                         device=input1.device)

    if inplace:
        assert input1.shape == output.shape, "inplace operation requires input and output tensors to be of the same shape"

    full_hash = full_hash_seed()

    grid = lambda META: (triton.cdiv(n_size * p_size, META['BLOCK_SIZE_N'] *
                                     META['BLOCK_SIZE_P']), full_hash)

    rand_hash = full_hash

    dropout_prob = p
    approximate_gelu = approximate

    with torch.cuda.device(input1.device.index):
        fused_bmm_dropout_gelu_kernel[grid](
            input1,
            input2,
            output,
            input1.stride(0),
            input1.stride(1),
            input1.stride(2),
            input2.stride(0),
            input2.stride(1),
            input2.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            batch_size,
            n_size,
            m_size,
            p_size,
            dropout_prob,
            training,
            inplace,
            approximate_gelu,
            rand_hash=rand_hash,
        )
    return output
