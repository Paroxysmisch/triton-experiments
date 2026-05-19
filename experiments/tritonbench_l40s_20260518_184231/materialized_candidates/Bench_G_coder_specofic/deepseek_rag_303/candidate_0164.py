import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform as plat
from vllm.utils import prod

from .gaussian_noise import gaussian_noise
from .snr import estimate_snr
from .tanh import tanh
from .relu import relu
from .gelu import gelu
from .fast_gelu import fast_gelu
from .activation_fn_enum import ActivationFn


def solve_by_configs(heuristics, configs, key, prune):
    configs = sorted(configs, key=lambda x: plat.get_pre_alloc_size(x))
    minimal_configs = [solve(heuristics, key, prune, init=x) for x in configs]
    all_valid = any(minimal_configs)
    if not all_valid:
        return None
    min_idx = min(
        (x.cost, i) for i, x in enumerate(minimal_configs) if x is not None
    )[1]
    return configs[min_idx]


def solve(heuristics, key, prune, init=None):
    def counter_key():
        return "counter-" + "-".join(str(x) for x in heuristics)

    def counter():
        x = plat.get(counter_key())
        if x is None:
            plat.set(counter_key(), 0)
        return plat.get(counter_key())

    def solve_recursive(heuristics, key, counter, prune, depth, init):
        log2 = lambda x: math.log(x) / math.log(2)
        if log2(counter + 1) + depth / 2 > plat.config().max_autotune_depth:
            return None

        if counter >= prune:
            return None

        plat.set(counter_key(), counter + 1)
        v_best = None
        h_best = None
        for h in heuristics:
            config = h.apply(key, init)
            h.counter = counter
            h.num_warps = plat.autotune_num_warps(config)
            plat.record_autotune_meta(config)
            v = plat.efficiency_for(config)
            if v_best is None or v > v_best:
                v_best = v
                h_best = h
            elif v == v_best:
                h_best = None
            h.counter -= 1

        if h_best is not None:
            if h_best.num_stages == 0:
                result = h_best.config
            else:
                sub_heuristics = [
                    Heuristic(h_best.kernel, h_best.num_stages - 1)
                    for _ in range(plat.autotune_num_warps(h_best.config))
                ]
                result = solve_by_configs(
                    sub_heuristics, plat.best_configs_for(h_best.config), key, prune
                )
            plat.set(counter_key(), 0)
        else:
            result = None

        return result

    return solve_recursive(
        heuristics, key, counter(), prune, plat.config().max_autotune_depth, init
    )


UNROLL = 128 * 8


@triton.jit
def kernel_fma(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    n_blocks,
    K,
    cs_batch: tl.constexpr,
    cs_batch_stride: tl.constexpr,
    cs_block: tl.constexpr,
    cs_block_stride: tl.constexpr,
    cs_k: tl.constexpr,
    cs_k_stride: tl.constexpr,
    cs_n: tl.constexpr,
    cs_n_stride: tl.constexpr,
    cs_out: tl.constexpr,
    cs_out_stride: tl.constexpr,
    CACHE_KEY_MASK: tl.constexpr,
    activation_fn_enum: tl.constexpr,
    save_pre_activation: tl.constexpr,
    pre_act_out_ptr,
    gaussian_noise_mean: tl.constexpr,
    gaussian_noise_std_dev: tl.constexpr,
):
    cs_batch_idx = tl.program_id(0)
    cs_block_idx = tl.program_id(1)
    cs_k_idx = tl.program_id(2)

    x_ptr += cs_batch_idx * cs_batch_stride
    out_ptr += cs_batch_idx * cs_batch_stride
    if save_pre_activation:
        pre_act_out_ptr += cs_batch_idx * cs_batch_stride

    if cs_n_idx := (tl.arange(0, UNROLL) + cs_block_idx * cs_block)[:, None] < n_blocks:
        x_addr = x_ptr + cs_k_idx * cs_k_stride + cs_n_idx * cs_n_stride
        w_addr = w_ptr + cs_k_idx * cs_k_stride + cs_n_idx * cs_n_stride
        accum_ptr = tl.arange(0, UNROLL)

        if cs_k_idx < K:
            x = tl.load(x_addr, mask=cs_k_idx < K, other=0.0)
            w = tl.load(w_addr, mask=cs_k_idx < K).to(tl.float32)
        else:
            x = tl.zeros((UNROLL,), dtype=tl.float16)
            w = tl.zeros((UNROLL,), dtype=tl.float32)

        bias = tl.load(bias_ptr + cs_n_idx * cs_n_stride)

        if gaussian_noise_mean != 0.0 or gaussian_noise_std_dev != 1.0:
            gn = gaussian_noise(
                mean=gaussian_noise_mean, std_dev=gaussian_noise_std_dev, size=UNROLL
            )
            x = x + gn

        if cs_out_idx := (
            tl.arange(0, UNROLL) + cs_block_idx * cs_block
        )[:, None] < cs_out:
            key_ptrs = accum_ptr + cs_out_idx * cs_out_stride
            value_ptrs = x.to(tl.float32) * w
            # tl.store(key_ptrs, value_ptrs, mask=cs_out_idx<cs_out)
            tl.atomic_add(key_ptrs, value_ptrs, mask=cs_out_idx < cs_out)
            # At last add bias
            value = tl.load(cs_out_idx * cs_out_stride + accum_ptr + bias)
        else:
            accum = tl.zeros((UNROLL,), dtype=tl.float32)
            value = accum + bias

        if save_pre_activation:
            # Save pre-activation results
            pre_act_out = tl.where(cs_out_idx, value, 0.0)
            tl.store(
                pre_act_out_ptr + accum_ptr, pre_act_out, mask=cs_out_idx.to(tl.float32)
            )

        # Do activation before adding result to final output
        if activation_fn_enum == ActivationFn.TANH:
            value = tanh(value)
        elif activation_fn_enum == ActivationFn.RELU:
            value = relu(value)
        elif activation_fn_enum == ActivationFn.GELU:
            value = gelu(value)
        elif
