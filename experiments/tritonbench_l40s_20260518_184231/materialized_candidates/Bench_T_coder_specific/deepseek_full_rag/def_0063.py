import torch
import triton
import triton.language as tl

@triton.jit()
def _xformers_tiled_matmul_kernel(
    A11, A12, A13, A21, A22, A23, A31, A32, A33,
    B11, B12, B13, B21, B22, B23, B31, B32, B33,
    C11, C12, C13, C21, C22, C23, C31, C32, C33,
    M1, M2, M3, N1, N2, N3, K1, K2, K3,
    stride_am1, stride_am2, stride_am3, stride_ak1, stride_ak2, stride_ak3,
    stride_bk1, stride_bk2, stride_bk3, stride_bn1, stride_bn2, stride_bn3,
    stride_cm1, stride_cm2, stride_cm3, stride_cn1, stride_cn2, stride_cn3,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr
):
    # Kernel logic for matrix multiplication

def _launch_triton_matmul(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
    c: List[List[torch.Tensor]],
    ms: List[int],
    ns: List[int],
    ks: List[int],
) -> None:
    strides_am, strides_ak = _get_strides(a, "first operand", "m", "k")
    strides_bk, strides_bn = _get_strides(b, "second operand", "k", "n")
    strides_cm, strides_cn = _get_strides(c, "output", "m", "n")

    ACC_TYPE = (
        tl.float32
        if c[0][0].dtype in [torch.float16, torch.bfloat16, torch.float32]
        else tl.int32
    )

    def grid(META):
        return (
            sum(triton.cdiv(m, META["BLOCK_M"]) for m in ms)
            * sum(triton.cdiv(n, META["BLOCK_N"]) for n in ns),
            META["SPLIT_K"],
        )

    _xformers_tiled_matmul_kernel[grid](
        *[a[min(i, len(a) - 1)][min(j, len(a[0]) - 1)] for i in range(3) for j in range(3)],
        *[b[min(i, len(b) - 1)][min(j, len(b[0]) - 1)] for i in range(3) for j in range(3)],
        *[c[min(i, len(c) - 1)][min(j, len(c[0]) - 1)] for i in range(3) for j in range(3)],
        *[ms[i] if len(ms) > i else 0 for i in range(3)],
        *[ns[i] if len(ns) > i else 0 for i in range(3)],
        *[ks[i] if len(ks) > i else 0 for i in range(3)],
        *strides_am, *strides_ak, *strides_bk, *strides_bn, *strides_cm, *strides_cn,
        ACC_TYPE=ACC_TYPE,
    )
