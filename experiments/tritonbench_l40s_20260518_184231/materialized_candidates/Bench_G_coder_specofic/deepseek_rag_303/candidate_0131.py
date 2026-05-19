import triton
import triton.language as tl
from enum import Enum
from packaging import version

_triton3 = version.parse(triton.__version__) >= version.parse("3.0.0")


class Activation(Enum):
    LeakyReLU = "leaky_relu"


def _get_kernel_config(name: str, args, key):
    triton_config = None
    _kernel_name = _get_qualified_kernel_name(name)
    if hasattr(tl, "config_of_kind") and _triton3:
        triton_config = tl.config_of_kind(key, {_kernel_name})
    if not triton_config:
        triton_config = KernelConfigManager.retrieve_config(key, _kernel_name, args)
    return triton_config


def _get_grid(args, config) -> tuple:
    if config and config.reduction_hints is not None:
        if config.reduction_hints.write_hint:
            return (1, 1, 1)
    M, N, K = args[1], args[2], args[3]
    BLOCK_M = BLOCK_N = min(config.preferred_block_sizes[0], 65536 // 16) if config else 128
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    return grid_m, grid_n


def matmul_kernel(config=None):
    launch_args_hook = "triton_kernels.matmul.launch_args"
    pre_hook = None
    if config and config.autotune:
        key = ("f32", "f32", "f32")
        pre_hook = lambda args: KernelConfigManager.record_launch_args(key, args, launch_args_hook)
    key = ("f32", "f32", "f32")
    triton_config = _get_kernel_config(__name__, key, key)
    if not triton_config:
        triton_config = KernelConfigManager.retrieve_config(key, _get_qualified_kernel_name(__name__), key)
    if config and config.reduction_hints:
        if config.activation == Activation.LeakyReLU:
            config.reduction_hints.write_hint = None
    kernel_settings = _get_kernel_settings(config, triton_config)
    grid = _get_grid(key, triton_config)

    @triton.jit(key, **kernel_settings)
    def kernel(C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        _mm_iomut_check_init()
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = kernel_utils.mul(accumulator, accumulator)
        if config and config.activation == Activation.LeakyReLU:
            c = kernel_utils.leaky_relu(c)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c)

    return kernel, pre_hook, grid


def matmul(a: torch.Tensor, b: torch.Tensor, config: KernelConfig = None, activation: Optional[Union[str, Activation]] = None) -> torch.Tensor:
    if activation is not None:
        if isinstance(activation, str):
            activation = Activation[activation]
        config = config or KernelConfig()
        config.activation = activation

    shape_a = a.shape
    shape_b = b.shape
    assert len(shape_a) == 2
    assert len(shape_b) == 2
    M, K = shape_a
    K, N = shape_b
    assert K == shape_a[1] == shape_b[0], "Incompatible dimensions"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    kernel, hook, grid = matmul_kernel(config)
    stream = torch.cuda.current_stream()

    a_ = a.view((M * K, 1))
    b_ = b.view((1, N * K))

    cublas_handle = get_cublas_handle()
    # Allocate workspace.
    workspace_size, workspace_size_fused = 0, 0
    if config and config.autotune:
        workspace_size = get_max_workspace_size(cublas_handle)
        workspace_size_fused = 2 * workspace_size

    workspace = torch.empty((workspace_size,), dtype=a.dtype, device=a.device)
    workspace_fused = torch.empty(
        (workspace_size_fused,), dtype=a.dtype, device=a.device
    )
    kernel[(grid[0], grid[1])](
        c,
        a_,
        b_,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        a_.stride(0),
        a_.stride(1),
        b_.stride(0),
        b_.stride(1),
        post_hook=lambda: hook((c, a_, b_, M, N, K)),
    )
    reset_all_seeds()

    return c
