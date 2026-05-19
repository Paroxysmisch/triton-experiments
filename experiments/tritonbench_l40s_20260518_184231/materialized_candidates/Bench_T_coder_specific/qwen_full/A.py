import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

def generate_wrapper(func_name: str, signature: str, const_args: str, backend: str, device: str):
    wrapper = f"def {func_name}({signature}):\n"
    wrapper += f"    args = [A, B]\n"
    wrapper += "    for idx, arg in enumerate(args):\n"
    wrapper += "        if isinstance(arg, torch.Tensor):\n"
    wrapper += "            args[idx] = arg\n"
    wrapper += "        else:\n"
    wrapper += "            args[idx] = torch.tensor(arg)\n"
    wrapper += f"    if {const_args}:\n"
    wrapper += "        A, B = args\n"
    wrapper += "        return torch.{func_name}(A, B, left={left}, out={out})\n"
    wrapper += "    else:\n"
    wrapper += "        A, B = args\n"
    wrapper += "        return torch.{func_name}(A, B, left={left})\n"
    return wrapper

@triton.jit
def linalg_solve_kernel(
    A,
    B,
    stride_za,
    stride_ha,
    stride_ma,
    stride_ak,
    stride_zb,
    stride_hb,
    stride_mb,
    stride_kb,
    stride_zx,
    stride_hx,
    stride_mx,
    n,
    k,
    ONE,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LEFT: tl.constexpr,
):
    """
    Compute the solution of a square system of linear equations with a unique solution.
    Args:
        A: The input matrix A of shape (n, n) or (batch, n, n)
        B: The input matrix B of shape (n, 1) or (batch, n, 1)
        stride_za, stride_ha, stride_ma, stride_ak: Stride for accessing A
        stride_zb, stride_hb, stride_mb, stride_kb: Stride for accessing B
        stride_zx, stride_hx, stride_mx: Stride for accessing X
        n: Size of the square matrix
        k: Dummy parameter for kernel reuse
        ONE: Tensor of value 1 to allow for division
        BLOCK_SIZE_M: Block size for the inner most loop
        BLOCK_SIZE_N: Block size for the middle loop
        BLOCK_SIZE_K: Block size for the outer loop
        GROUP_SIZE_M: Group size for the inner most loop
        LEFT: Whether to solve AX = B or XA = B
    Returns:
        X: The solution matrix of shape (n, 1) or (batch, n, 1)
    """
    # Program ids
    pid_z = tl.program_id(axis=0)
    if len(A.shape) == 4:
        pid_h = tl.program_id(axis=1)
        A += pid_z * stride_za + pid_h * stride_ha
        B += pid_z * stride_zb + pid_h * stride_hb
    else:
        A += pid_z * stride_za
        B += pid_z * stride_zb
    # Create thread ids
    #拓扑排序
    # Inner most loop tile
    tile_m = tl.arange(0, BLOCK_SIZE_M)
    tile_n = tl.arange(0, BLOCK_SIZE_N)
    # Middle loop tile
    tile_k = tl.arange(0, BLOCK_SIZE_K)
    # Loop over k
    for kk in range(0, k, BLOCK_SIZE_K):
        # Create range for accessing memory
        rk = kk + tile_k
        # Load A tiles
        a = tl.load(
            A + (tile_m[:, None] * stride_ma + rk[None, :] * stride_ak),
            mask=(tile_m[:, None] < n) & (rk[None, :] < k),
            other=0.0,
        )
        if LEFT:
            # Load B tiles
            b = tl.load(
                B + (tile_m[:, None] * stride_mb + tile_n[None, :] * stride_kb),
                mask=(tile_m[:, None] < n) & (tile_n[None, :] < 1),
                other=0.0,
            )
            # Matmul
            c = tl.dot(a, b, allow_tf32=False)
            # Store result
            tl.store(
                B + (tile_m[:, None] * stride_mx + tile_n[None, :] * stride_k),
                c,
                mask=(tile_m[:, None] < n) & (tile_n[None, :] < 1),
            )
        else:
            # Load B tiles
            b = tl.load(
                B + (tile_n[:, None] * stride_kb + rk[None, :] * stride_mb),
                mask=(tile_n[:, None] < n) & (rk[None, :] < k),
                other=0.0,
            )
            # Matmul
            c = tl.dot(b, a, allow_tf32=False)
            # Store result
            tl.store(
                B + (tile_n[:, None] * stride_mx + tile_m[None, :] * stride_k),
                c,
                mask=(tile_n[:, None] < n) & (tile_m[None, :] < 1),
            )
    # Compute 1/k
    k_rec = tl.math.rsqrt(kk + 1)
    # Normalize
    mask = tile_m[:, None] < n
    x = tl.load(
        B + tile_m[:, None] * stride_mx, mask=mask, other=0.0
    )  # Loading stored result
    x *= ONE * k_rec
    tl.store(
        B + tile_m[:, None] * stride_mx, x, mask=mask
    )  # Overwriting stored result

def get_kernel_signature(args, kwargs):
    sig = ""
    for a in args:
        if isinstance(a, torch.Tensor):
            sig += f"{a.ndim} "
        else:
            sig += "0 "
    sig = sig.strip()
    const_args = False
    if kwargs.get("left") is not None:
        const_args = True
        sig += " 1"
    else:
        sig += " 0"
    if kwargs.get("out") is not None:
        const_args = True
        sig += " 1"
    else:
        sig += " 0"
    return sig, const_args

func_name = "linalg_solve"
backend = "CUDA"
device = "CUDA"
const_args, left, out = False, False, False
signature, const_args = get_kernel_signature([torch.randn(3, 3, device="cuda"), torch.randn(3, 1, device="cuda")], {})
wrapper = generate_wrapper(func_name, signature, const_args, backend, device)
print(wrapper)
