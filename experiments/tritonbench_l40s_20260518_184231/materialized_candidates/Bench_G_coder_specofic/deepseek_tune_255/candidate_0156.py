import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK"],
)
@triton.heuristics(
    {
        "IS_FP16": lambda args: args["x"].dtype == torch.float16,
    }
)
@triton.jit
def _softmax(
    x,
    o,
    BT: tl.constexpr,
    BK: tl.constexpr,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # Triton kernel for softmax
    BM = 32 // tl.next_power_of_2(BK)
    BK = min(BK, tl.cdiv(32, BM))

    bh, bt = tl.program_id(0), tl.program_id(1)

    m_prev = tl.full([BM], float("-inf"), dtype=tl.float32)
    m = tl.full([BM], float("-inf"), dtype=tl.float32)

    mask = tl.where(
        tl.arange(0, BM) < (BM - BT),
        0,
        tl.full([BM], 1, dtype=tl.int32),
    ).to(tl.float32)

    for i in range(0, bt, BK):
        k = i + tl.arange(0, BK)
        p = tl.where(k < BT, -1e9, 0.0) if MASK_TYPE == "lower" else 0.0
        p = tl.where(k < BT, p, 1e9) if MASK_TYPE == "upper" else p

        if CAUSAL:
            p = tl.where(k <= BT // 2, p, 0.0)
            p = tl.where(k + BT // 2 < BT, p, 1e9)

        p = p + tl.load(x + bh * BT * BK + k, mask=k < BT, other=0.0)
        m_prev = tl.maximum(m_prev, p)
        p = tl.exp(p - m_prev)

        if LOG:
            o_prev = tl.where(k < BT, tl.load(x + bh * BT * BK + k, mask=k < BT, other=0.0), 0.0)
            o_prev = o_prev - m_prev
            o_prev = o_prev.to(x.dtype.element_ty)
            tl.store(o + bh * BT * BK + k, o_prev, mask=k < BT)
        else:
            tl.store(o + bh * BT * BK + k, p, mask=k < BT)

        m = tl.maximum(m, p)

    m = tl.max(m, axis=0)
    m_prev = tl.max(m_prev, axis=0)

    for i in range(0, bt, BK):
        k = i + tl.arange(0, BK)
        if CAUSAL and (BT // 2 <= i < bt - BT // 2):
            p = tl.where(k + BT // 2 < bt, 1e9, 0.0)
        else:
            p = 0.0

        p = p + tl.load(x + bh * BT * BK + k, mask=k < BT, other=0.0)
        p = tl.exp(p - m)

        if not LOG:
            p = p / (tl.exp(m - m_prev) + tl.sum(p, axis=0))

        tl.store(o + bh * BT * BK + k, p, mask=k < BT)


def softmax(x, log=False, dim=-1, causal=False, mask_type=None):
    assert dim in [-1, len(x.shape) - 1], "dim must be -1 or the last dim"
    assert x.dim() >= 2, "input must be at least a 2D tensor"
    assert not x.is_floating_promotable(), "underflow in data type"

    BT = triton.next_power_of_2(x.shape[dim])
    BK = triton.next_power_of_2(32 // BT)

    o = torch.empty_like(x)

    CAUSAL = 0 if causal is False else 1
    MASK_TYPE = "lower" if mask_type is "lower" else "upper" if mask_type is "upper" else None

    grid = lambda META: (x.shape[0] // META["BS"], x.shape[0] % META["BS"])

    _softmax[grid](
        x,
        o,
        BT=BT,
        BK=BK,
        LOG=1 if log else 0,
        CAUSAL=CAUSAL,
        MASK_TYPE=MASK_TYPE,
    )

    return o


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK"],
)
@triton.heuristics(
    {
        "IS_FP16": lambda args: args["out"].dtype == torch.float16,
    }
)
@triton.jit
def _softmax_backward(
    out,
    dout,
    din,
    BT: tl.constexpr,
    BK: tl.constexpr,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # Triton kernel for softmax backward
    BM = 32 // tl.next_power_of_2(BK)
    BK = min(BK, tl.cdiv(32, BM))

    bh, bt = tl.program_id(0), tl.program_id(1)

    mask = tl.where(
        tl.arange(0, BM) < (BM - BT),
        0,
        tl.full([BM], 1, dtype=tl.int32),
    ).to(tl.float32)

    out = tl.load(out + bh * BT * BK + tl.arange(0, BK), mask=tl.arange(0, BK) < BT, other=0.0)

    if LOG:
        dout = tl.load(dout + bh * BT * BK + tl.arange(0, BK), mask=tl.arange(0, BK) < BT, other=0.0)
        din = dout * (out - tl.exp(out))
    else:
        dout = tl.load(dout + bh * BT * BK + tl.arange(0, BK), mask=tl.arange(0, BK) < BT, other=0.0
