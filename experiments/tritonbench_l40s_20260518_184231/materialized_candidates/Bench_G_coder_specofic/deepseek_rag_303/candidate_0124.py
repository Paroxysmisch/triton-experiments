import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

HAS_SCALE = triton.has_feature("fp16") or triton.has_feature("bf16")

if TRITON3:
    from functools import cmp_to_key
    from triton.runtime.table import param_pruner

    def backoff_full(b):
        return triton.cdiv(b, 0) == 0

else:
    from triton.runtime.grid import backoff_full

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 32}, num_warps=1),
        triton.Config({"BLOCK": 32}, num_warps=2),
        triton.Config({"BLOCK": 32}, num_warps=4),
        triton.Config({"BLOCK": 32}, num_warps=8),
        triton.Config({"BLOCK": 32}, num_warps=16),
        triton.Config({"BLOCK": 32}, num_warps=32),
    ],
    key=[triton.key.NUM_CRITICAL_PATHS, triton.key.CONGESTION],
    prune_configs_by={
        "early_config_prune": triton.config_pruner.EarlyConfigPruner(),
        "perf_cache": "perf.cache",
    },
    warmup=1000 if not TRITON3 else 4000,
    rep=1000 if not TRITON3 else 4000,
    timeout=20 if not TRITON3 else 5,
    reset_to_zero=["b_m"],
)
@triton.heuristics({"HAS_SCALE": lambda args: args["x"].dtype in [tl.float16, tl.bfloat16]})
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    i_n,
    i_d,
    o_d,
    m_d,
    b_m,
    N: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # indices
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    o_d = tl.arange(0, B)
    m_d = tl.arange(0, B)
    # partial sums
    b_x = tl.load(
        x + (i_n * N + o_d[:, None] * B + i_d[None, :])
    )  # NN,11,1-1024:3
    b_m = tl.max(b_x, axis=0)  # 11,1
    if HAS_SCALE:
        b_x = (tl.math.exp((b_x - b_m[:, None]) / 2) * 2).to(x.dtype)
    else:
        b_x = tl.math.exp(b_x - b_m[:, None])  # 11,11
    # log(sum(exp(A))) = log(sum(x(i) * exp(x(i) - c))) = log(x(m) * sum(exp(x(i) - x(m) - c)))
    # = log(x(m)) + log(sum(exp(x(i) - x(m))))
    xm = tl.max(b_x)
    b_x = tl.math.log(tl.sum(tl.exp(b_x - xm))) + b_m + xm
    tl.store(z + (i_n * N + o_d[:, None] * B + i_d[None, :]), b_x)

def logsumexp_fwd(x, bench=False):
    N, D, B = x.shape
    block = min(1024, triton.next_power_of_2(D))
    shape = (N, triton.cdiv(D, block))
    z = x.new_empty(N, B)
    i_d = x.stride(0)
    x_d = x.stride(1)
    # invariants: x, block, i_d, N, block
    kernel_init = logsumexp_fwd_kernel[shape]
    if bench:
        print("INVARIANTS:", kernel_init.num_invocations, kernel_init.invocation_shapes)
        return
    else:
        grid = (int(N), int(D))
    kernel_init[(grid[0] * grid[1],)]
    z = z.sum(dim=1, keepdim=True)
    return z.to(x.dtype)
