import math
import torch
import triton
import triton.language as tl
from models.mamba.ops.triton.softplus import softplus

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 1}),
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    batch, seqlen, nheads, chunk_size, dt_min, dt_max,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_dt_out_batch, stride_dt_out_chunk,
    stride_dt_out_head, stride_dt_out_csize, stride_dA_cs_batch,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    DT_SOFTPLUS: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = softplus(dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_H": 1}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 2}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 4}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, dt_bias_ptr, ddt_ptr,
    dA_ptr, ddt_bias_ptr, batch, seqlen, nheads, chunk_size,
    dt_min, dt_max, stride_ddA_batch, stride_ddA_chunk,
    stride_ddA_head, stride_ddA_csize, stride_ddt_out_batch,
    stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_ddt_batch, stride_ddt_seqlen,
    stride_ddt_head, stride_dA_head, stride_ddt_bias_head,
    DT_SOFTPLUS: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * chunk_size * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (
        offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize
    )
    ddA_ptrs = ddA_ptr + (
        offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize
    )
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    ddt_ptrs = ddt_ptr + (
        offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    ddA = tl.load(
        ddA_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    ddt_out = tl.load(
        ddt_out_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = softplus(dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    ddt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0
    )
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(
        ddt_ptrs,
        ddt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
    )
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(
            ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads
        )

def _chunk_cumsum_fwd(
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    grid_chunk_cs = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt,
            A,
            dt_bias,
            dt_out,
            dA_cumsum,
            batch,
            seqlen,
            nheads,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0),
            dt_out.stride(2),
            dt_out.stride(1),
            dt_out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out

def _chunk_cumsum_bwd(
    ddA,
    ddt_out,
    dt,
    A,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    ddt=None,
):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    else:
        ddt_bias = None
    if ddt is not None:
        assert ddt.shape == dt.shape
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    grid_chunk_cs = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](
            ddA,
            ddt_out,
            dt,
            A,
            dt_bias,
            ddt,
            dA,
            ddt_bias,
            batch,
            seqlen,
            nheads,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            ddA.stride(0),
            ddA.stride(2),
            ddA.stride(1),
            ddA.stride(3),
            ddt_out.stride(0),
            ddt_out.stride(2),
            ddt_out.stride(1),
            ddt_out.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            ddt.stride(0),
            ddt.stride(1),
            ddt.stride(2),
            dA.stride(0),
            ddt_bias.stride(0) if ddt_bias is not None else 0,
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return ddt, dA, ddt_bias

class ChunkStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if (
            x.stride(-1) != 1 and x.stride(1) != 1
        ):  # Either M or K dimension should be contiguous
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
        dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, ngroups=ngroups)
        dB = dB.to(B.dtype)
        return dB, dx, ddt, ddA_cumsum, None

def chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
    return ChunkStateFn.apply(B, x, dt, dA_cumsum, states_in_fp32)
