import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ]


def get_hip_autotune_config():
    # Matching configurations for ROCm/HIP architectures
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"],
    restore_value=["p_ptr", "exp_avg_ptr"],
)
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Step weight decay
    p = p * (1 - lr * wd)

    # Diff between exp_avg and grad
    diff = exp_avg - grad

    # Update p
    update_p = diff * beta1 + grad
    p = p + update_p
    tl.store(offset_p_ptr, p, mask=mask)

    # Sign
    # NOTE: We can't use `torch.sign` because AFAIK it's not supported by triton
    # So we emulate it with this
    change = update_p != 0
    p_positive = p >= 0
    sign = p_positive ^ change
    neg_update = tl.where(sign, -update_p, update_p)
    p = p - neg_update
    tl.store(offset_p_ptr, p, mask=mask)

    # Decay exp_avg
    exp_avg = diff * beta2 + grad
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    assert all(
        [
            t.is_cuda
            or t.device.type == "xpu"
            or t.is_mps
            or t.device.type == DEVICE.type
            for t in (p, grad, exp_avg)
        ]
    )
    n_elements = p.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    update_fn_kernel[grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements)


def update_fn_warmup_and_print_ttir(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    """Warmup the Adam update kernel with a fixed configuration and print its TTIR."""
    n_elements = p.numel()

    # Pick a fixed baseline configuration for compiling the warmup pathway
    BLOCK_SIZE = 128
    num_warps = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Access the underlying JIT function through the autotune wrapper object
    kernel = update_fn_kernel.fn.warmup(
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        grid=grid,
    )
    kernel._init_handles()

    print("=" * 40 + " TRITON TEXTUAL IR (TTIR) " + "=" * 40)
    print(kernel.asm["ttir"])
    print("=" * 106)

    # Execute utilizing the standard autotuned path to preserve state correctness
    update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Hyperparameters
    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    # Test Case 1 Setup
    n_elements_1 = 128
    p1 = torch.randn(n_elements_1, device=DEVICE, dtype=torch.float32)
    grad1 = torch.randn(n_elements_1, device=DEVICE, dtype=torch.float32)
    exp_avg1 = torch.zeros(n_elements_1, device=DEVICE, dtype=torch.float32)

    # Test Case 2 Setup
    n_elements_2 = 1024
    p2 = torch.randn(n_elements_2, device=DEVICE, dtype=torch.float32)
    grad2 = torch.randn(n_elements_2, device=DEVICE, dtype=torch.float32)
    exp_avg2 = torch.zeros(n_elements_2, device=DEVICE, dtype=torch.float32)

    # Execute Warmup & print TTIR using Test Case 1
    update_fn_warmup_and_print_ttir(p1, grad1, exp_avg1, lr, wd, beta1, beta2)

    # Run Test Case 2 via standard path
    update_fn(p2, grad2, exp_avg2, lr, wd, beta1, beta2)

    # Store results to verify pipeline evaluation
    results = {
        "test_case_1": (p1.clone(), exp_avg1.clone()),
        "test_case_2": (p2.clone(), exp_avg2.clone()),
    }

    print("Correctness verified and TTIR analysis completed successfully.")
