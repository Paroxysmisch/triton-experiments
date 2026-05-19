import torch
import triton
import triton.language as tl
from flag_gems.utils.tools import estimate_max_grid_size
from flag_gems.utils.cuda_utils import get_device_capability


def config_prune_fn(configs):
    cap = get_device_capability()
    max_block_size = triton.next_power_of_2(cap[0] * cap[1])
    return [config for config in configs if config["BLOCK_SIZE"] <= max_block_size]


def determine_num_warps(per_warp_size):
    num_warps = 4
    if per_warp_size >= 2048:
        num_warps = 8
    if per_warp_size >= 4096:
        num_warps = 16
    return num_warps


def generate_configs():
    block_sizes = [128, 256, 512, 1024, 2048, 4096]
    warps = [1, 2, 4, 8, 16, 32]
    configs = []
    for warp in warps:
        for size in block_sizes:
            configs.append({"num_warps": warp, "BLOCK_SIZE": size})
    return configs


@triton.autotune(
    configs=config_prune_fn(generate_configs()),
    key=["n", "divisible_by_16_n"],
)
@triton.heuristics({
    "num_warps": lambda args: determine_num_warps(args["BLOCK_SIZE"]),
})
@triton.jit
def qr_kernel_determinant_reduced_mode(
    input_ptr,
    tau_ptr,
    output_ptr,
    n,
    divisible_by_16_n,
    # strides
    stride_input_batch,
    stride_input_t,
    stride_input_f,
    stride_tau_batch,
    stride_tau_m,
    stride_tau_k,
    stride_output_batch,
    stride_output_row,
    stride_output_col,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_r = tl.program_id(axis=2)
    # The last division by 16 is just for a potential future implementation hint to runtime
    # that the codegen could benefit from knowing that certain loops are divisible by 16.
    pid_f = tl.program_id(axis=0) * 16 + (tl.arange(0, 16) % 4)
    # A block will never cross a batch or time boundary.
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n, n),
        strides=(stride_input_t, stride_input_f),
        offsets=(pid_r * BLOCK_SIZE, pid_f),
        block_shape=(BLOCK_SIZE, 4),
        order=(1, 0),
    )
    tau_block_ptr = tl.make_block_ptr(
        base=tau_ptr,
        shape=(n, 1),
        strides=(stride_tau_t, stride_tau_f),
        offsets=(pid_r * BLOCK_SIZE, pid_f),
        block_shape=(BLOCK_SIZE, 1),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n, n),
        strides=(stride_output_t, stride_output_f),
        offsets=(pid_b * BLOCK_SIZE, pid_f),
        block_shape=(BLOCK_SIZE, 1),
        order=(1, 0),
    )

    # load f vector
    f0 = tl.load(tau_block_ptr, boundary_check=(0,), padding_option="zero")

    fn = tl.load(
        input_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    ).to(tl.float32)

    # GEMV: Multiply-Accumulate
    acc0 = tl.zeros((BLOCK_SIZE,), dtype=f0.dtype)
    for _ in range(0, tl.cdiv(n, 4)):
        fn4 = tl.load(
            input_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)
        acc0 += f0 * fn4[:, 0]
        f0 = fn4[:, 1]
        input_block_ptr = tl.advance(input_block_ptr, [0, 4])

    acc0 = acc0.to(input_ptr.dtype.element_ty)

    # Store acc directly into the output
    tl.store(output_block_ptr, acc0[:, None])


def determinant_via_qr_triton(a, mode: str = "reduced", out=None):
    assert mode == "reduced"
    batch, n, _ = a.shape
    if a.stride(-1) != 1:
        a = a.contiguous()
    capability = get_device_capability()
    max_grid = estimate_max_grid_size(
        capability,
        {
            "n": n,
            "divisible_by_16_n": n % 16 == 0,
            "BLOCK_SIZE": 128,
        },
    )
    if out is None:
        out = torch.empty(batch, n, n, device=a.device, dtype=torch.float)
    else:
        out = out.contiguous()
    grid = (batch, triton.cdiv(n, 16), 1)
    signature = "*fp64,i32,i32,*fp64,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32->*fp64"

    qr_kernel_determinant_reduced_mode[grid](
        a,
        a,
        out,
        n,
        n % 16 == 0,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=128,
        num_stages=3,
        num_warps=8,
    )
    return out
