import torch
import triton
import triton.language as tl
from torch.fft import _utils
from flag_gems.utils.shape_utils import volume

def _signature(out, input, s, dim, norm):
    return (out, input, s, dim, norm)

@triton.jit
def _rfftn_forward(x, s, norm, x_stride, x_batch_stride, x_last_stride, output, b, d, n,
                   OUTPUT_TYPE: tl.constexpr, X_STRIDED: tl.constexpr, BATCHED: tl.constexpr,
                   STRIDED_BATCH: tl.constexpr, CONJUGATE: tl.constexpr, NORM_NEEDED: tl.constexpr,
                   LAST_DIM_POW2: tl.constexpr, MIDDLE_DIM_POW2: tl.constexpr,
                   PRECOMPUTE_PHASE: tl.constexpr, REVERSE_FIRST_PASS: tl.constexpr,
                   ALLOW_TF32: tl.constexpr):
    batch_pid = tl.program_id(0)
    middle_dim_pid = tl.program_id(1)
    first_pass_pid = tl.program_id(2)
    last_dim = n
    middle_dim = d
    if BATCHED:
        x_batch_stride = x_batch_stride.to(tl.int64)
        x_batch_offset = batch_pid * x_batch_stride
        tl.static_print("batch_pid", batch_pid, "x_batch_stride", x_batch_stride,
                        "x_batch_offset", x_batch_offset)
    else:
        x_batch_offset = 0
    if MIDDLE_DIM_POW2:
        x_middle_stride = tl.cdiv(middle_dim, 2) * 2 * x_last_stride
    else:
        x_middle_stride = middle_dim * x_last_stride
    x_middle_offset = middle_dim_pid * x_middle_stride
    tl.static_print("x_middle_stride", x_middle_stride, "x_middle_offset", x_middle_offset)
    if LAST_DIM_POW2:
        x_last_stride = 2 * x_last_stride
        x_last_offset = first_pass_pid * 2 * x_last_stride
    else:
        x_last_stride = x_last_stride
        x_last_offset = first_pass_pid * x_last_stride
    tl.static_print("x_last_stride", x_last_stride, "x_last_offset", x_last_offset)
    x_offset = x_batch_offset + x_middle_offset + x_last_offset
    tl.static_print("x_offset", x_offset)
    if X_STRIDED:
        x = tl.load(x + x_offset, eviction_policy='evict_last')
    else:
        x = tl.load(x + x_offset)
    tl.static_print("x", x)
    if CONJUGATE:
        x = tl.conj(x)
    if PRECOMPUTE_PHASE:
        if LAST_DIM_POW2:
            w = tl.exp(
                tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[:, None]
        else:
            w = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[:, None]
        if MIDDLE_DIM_POW2:
            if LAST_DIM_POW2:
                v = tl.exp(tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[None,
                                                                                               :]
            else:
                v = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[None, :]
        else:
            if LAST_DIM_POW2:
                v = tl.exp(tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[None,
                                                                                               :]
            else:
                v = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim))[None, :]
    else:
        if LAST_DIM_POW2:
            w = tl.exp(
                tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[:, None]
        else:
            w = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[:, None]
        if MIDDLE_DIM_POW2:
            if LAST_DIM_POW2:
                v = tl.exp(tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[None,
                                                                                                            :]
            else:
                v = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[None, :]
        else:
            if LAST_DIM_POW2:
                v = tl.exp(tl.arange(0, 2 * last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[None,
                                                                                                            :]
            else:
                v = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (-2j * tl.pi / last_dim)).to(OUTPUT_TYPE)[None, :]
    if LAST_DIM_POW2:
        if REVERSE_FIRST_PASS:
            x = tl.dot(w[1:last_dim // 2 + 1, :] * x, v[0:last_dim // 2 + 1, :])
        else:
            x = tl.dot(w[0:last_dim // 2 + 1, :] * x, v[0:last_dim // 2 + 1, :])
    else:
        if REVERSE_FIRST_PASS:
            x = tl.dot(w[0:last_dim, :] * x, v[0:last_dim, :])
        else:
            x = tl.dot(w[0:last_dim, :] * x, v[0:last_dim, :])
    if NORM_NEEDED:
        if LAST_DIM_POW2:
            x = x / last_dim
        else:
            x = x / last_dim
    if BATCHED:
        output_offset = batch_pid * b * d * n
    else:
        output_offset = 0
    output_offset = output_offset + middle_dim_pid * n * b + first_pass_pid * b + b // 2
    tl.static_print("output_offset", output_offset)
    tl.store(output + output_offset, x)

@triton.jit
def _ifftn_forward(x, s, norm, x_stride, x_batch_stride, x_last_stride, output, b, d, n,
                   OUTPUT_TYPE: tl.constexpr, X_STRIDED: tl.constexpr, BATCHED: tl.constexpr,
                   STRIDED_BATCH: tl.constexpr, CONJUGATE: tl.constexpr, NORM_NEEDED: tl.constexpr,
                   LAST_DIM_POW2: tl.constexpr, MIDDLE_DIM_POW2: tl.constexpr,
                   PRECOMPUTE_PHASE: tl.constexpr, REVERSE_FIRST_PASS: tl.constexpr,
                   ALLOW_TF32: tl.constexpr):
    batch_pid = tl.program_id(0)
    middle_dim_pid = tl.program_id(1)
    first_pass_pid = tl.program_id(2)
    last_dim = n
    middle_dim = d
    if BATCHED:
        x_batch_stride = x_batch_stride.to(tl.int64)
        x_batch_offset = batch_pid * x_batch_stride
        tl.static_print("batch_pid", batch_pid, "x_batch_stride", x_batch_stride,
                        "x_batch_offset", x_batch_offset)
    else:
        x_batch_offset = 0
    if MIDDLE_DIM_POW2:
        x_middle_stride = tl.cdiv(middle_dim, 2) * 2 * x_last_stride
    else:
        x_middle_stride = middle_dim * x_last_stride
    x_middle_offset = middle_dim_pid * x_middle_stride
    tl.static_print("x_middle_stride", x_middle_stride, "x_middle_offset", x_middle_offset)
    if LAST_DIM_POW2:
        x_last_stride = 2 * x_last_stride
        x_last_offset = first_pass_pid * 2 * x_last_stride
    else:
        x_last_stride = x_last_stride
        x_last_offset = first_pass_pid * x_last_stride
    tl.static_print("x_last_stride", x_last_stride, "x_last_offset", x_last_offset)
    x_offset = x_batch_offset + x_middle_offset + x_last_offset
    tl.static_print("x_offset", x_offset)
    if X_STRIDED:
        x = tl.load(x + x_offset, eviction_policy='evict_last')
    else:
        x = tl.load(x + x_offset)
    tl.static_print("x", x)
    if CONJUGATE:
        x = tl.conj(x)
    if PRECOMPUTE_PHASE:
        if LAST_DIM_POW2:
            w = tl.exp(
                tl.arange(0, 2 * last_dim).to(tl.float32) * (2j * tl.pi / last_dim))[:, None]
        else:
            w = tl.exp(tl.arange(0, last_dim).to(tl.float32) * (2j * tl.pi / last_dim))[:, None]
        if MIDDLE_DIM_POW2:
            if LAST_DIM_POW2:
                v = tl.exp(tl.arange(0, 2 * last_dim).to(tl.float32) * (
