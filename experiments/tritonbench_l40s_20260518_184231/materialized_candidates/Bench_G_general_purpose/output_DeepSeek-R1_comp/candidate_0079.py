import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A, B, absmax_inv,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    start_m = pid_m * GROUP_M * BLOCK_M
    start_n = pid_n * BLOCK_N

    block_ptr_A = tl.make_block_ptr(
        base=A,
        shape=(M, N),
        strides=(stride_am, stride_an),
        offsets=(start_m, start_n),
        block_shape=(GROUP_M * BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    a = tl.load(block_ptr_A, boundary_check=(0, 1))

    a_quant = (a * absmax_inv).to(tl.int8, saturate=True)
    a_transposed = tl.transpose(a_quant)

    block_ptr_B = tl.make_block_ptr(
        base=B,
        shape=(N, M),
        strides=(stride_bm, stride_bn),
        offsets=(start_n, start_m),
        block_shape=(BLOCK_N, GROUP_M * BLOCK_M),
        order=(1, 0),
    )
    tl.store(block_ptr_B, a_transposed, boundary_check=(0, 1))


def quantize_global_transpose(A: torch.Tensor):
    assert A.dim() == 2, "Input must be a 2D tensor."
    M, N = A.shape
    absmax = torch.max(torch.abs(A))
    absmax_inv = 127.0 / absmax if absmax != 0 else 0.0
    absmax_inv = absmax_inv.to(A.dtype).item()

    B = torch.empty((N, M), dtype=torch.int8, device=A.device)

    def grid(META):
        grid_m = (M + (META['GROUP_M'] * META['BLOCK_M']) - 1) // (META['GROUP_M'] * META['BLOCK_M'])
        grid_n = (N + META['BLOCK_N'] - 1) // META['BLOCK_N']
        return (grid_m * grid_n, )

    _quantize_global_transpose[grid](
        A, B, absmax_inv,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        GROUP_M=8,
    )
    return B
