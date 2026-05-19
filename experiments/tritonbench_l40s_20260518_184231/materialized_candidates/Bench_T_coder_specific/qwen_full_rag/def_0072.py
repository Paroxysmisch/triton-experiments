import torch
import triton
import triton.language as tl
from ..utils import calculate_settings_2d


@triton.jit
def conv2d_forward_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, stride, n_elements,
                          BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_OUT_FEAT: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
                          BLOCK_SIZE_SPATIAL: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(in_ptr1.shape[1], BLOCK_SIZE_OUT_FEAT)
    num_pid_n = tl.cdiv(in_ptr1.shape[2], BLOCK_SIZE_SPATIAL)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_n
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_OUT_FEAT + tl.arange(0, BLOCK_SIZE_BATCH))[
        None, :]
    offs_n = (pid_n * BLOCK_SIZE_SPATIAL +
              tl.arange(0, BLOCK_SIZE_IN_FEAT))[:, None]
    offs_k = tl.arange(0, BLOCK_SIZE_SPATIAL)[None, :]

    ram = tl.max_contiguous(tl.multiple_of(
        offs_m % in_ptr0.shape[0], BLOCK_SIZE_BATCH), BLOCK_SIZE_BATCH)
    rbn = tl.max_contiguous(offs_n % in_ptr0.shape[2], BLOCK_SIZE_IN_FEAT)
    rbk = tl.arange(0, BLOCK_SIZE_SPATIAL)
    rn = (pid_n * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL))

    a_ptrs = (in_ptr0 + (ram * stride +
                         rbn * stride + rbk)).to(tl.pointer_type(torch.float))
    b_ptrs = (in_ptr1 + (rbn * BLOCK_SIZE_SPATIAL +
                         rbk)).to(tl.pointer_type(torch.float))
    acc_tile = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_SPATIAL),
                        dtype=tl.float32)

    for k in range(0, tl.cdiv(in_ptr0.shape[1], BLOCK_SIZE_IN_FEAT)):
        selected_a = tl.load(a_ptrs)
        selected_b = tl.load(b_ptrs)
        acc_tile += tl.dot(selected_a, selected_b)
        a_ptrs += BLOCK_SIZE_IN_FEAT
        b_ptrs += (BLOCK_SIZE_SPATIAL * BLOCK_SIZE_IN_FEAT)

    acc_tile = acc_tile.to(tl.float16)

    c_ptrs = (out_ptr1 + offs_m * stride + rn)
    tl.store(c_ptrs, acc_tile)

    d_ptrs = (in_ptr2 + (ram * stride +
                         rn)).to(tl.pointer_type(torch.float))
    e_ptrs = (out_ptr2 + offs_m * stride + rn)
    mean = tl.load(d_ptrs)
    var = tl.load(d_ptrs + stride)

    z = (acc_tile - mean[:, None]) / tl.sqrt(var[:, None] + 1e-5)
    gamma_ptrs = (out_ptr3 + offs_m * stride + rn)
    beta_ptrs = (out_ptr3 + stride + offs_m * stride + rn)
    y = z * tl.load(gamma_ptrs) + tl.load(beta_ptrs)

    f_ptrs = (out_ptr2 + in_ptr1.shape[1] * stride + offs_m * stride + rn)
    tl.store(f_ptrs, y)


def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False) -> torch.Tensor:
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    assert input.shape[-2:] == weight.shape[-2:]
    assert weight.shape[1] == input.shape[1]
    assert weight.shape[0] == bias.shape[0] if bias is not None else True
    assert weight.shape[1] % groups == 0
    assert weight.shape[0] % groups == 0

    if groups != 1:
        raise Exception("Groups > 1 not supported")

    if running_mean is None:
        running_mean = torch.empty(
            weight.shape[0], device=input.device, dtype=torch.float32)
    if running_var is None:
        running_var = torch.empty(
            weight.shape[0], device=input.device, dtype=torch.float32)

    if bn_weight is None:
        bn_weight = torch.ones(weight.shape[0],
                               device=input.device, dtype=torch.float32)
    if bn_bias is None:
        bn_bias = torch.zeros(weight.shape[0],
                              device=input.device, dtype=torch.float32)

    in_feat_per_group = weight.shape[1] // groups
    out_feat_per_group = weight.shape[0] // groups

    mean = torch.empty((input.shape[0], ),
                       dtype=torch.float32, device="cuda")
    var = torch.empty((input.shape[0], ),
                      dtype=torch.float32, device="cuda")

    if training:
        cost_vol = torch.zeros([weight.shape[0]] +
                               list(input.shape[-2:]), device=input.device, dtype=torch.float16)
    else:
        cost_vol = torch.zeros([weight.shape[0]] +
                               list(input.shape[-2:]), device=input.device, dtype=torch.float32)

    conv_out = torch.empty_like(cost_vol)

    M = input.shape[0]
    N = weight.shape[0]
    K = input.shape[-2] * input.shape[-1]

    grid = lambda META: (
        triton.cdiv(M * N, META['BLOCK_SIZE_BATCH'] * META['BLOCK_SIZE_OUT_FEAT']) *
        triton.cdiv(K, META['BLOCK_SIZE_SPATIAL']), )

    pre_bn2 = torch.empty([input.shape[0], ] +
                          list(weight.shape[2:]), device=input.device, dtype=torch.float32)
    post_bn2 = torch.empty([input.shape[0], ] +
                           list(weight.shape[2:]), device=input.device, dtype=torch.float32)

    with torch.cuda.device(input.device.index):
        conv2d_forward_kernel[grid](
            input, weight, bias, cost_vol, pre_bn2, post_bn2,
            stride[0], M * N * K,
            BLOCK_SIZE_BATCH=M,
            BLOCK_SIZE_OUT_FEAT=out_feat_per_group,
            BLOCK_SIZE_IN_FEAT=in_feat_per_group,
            BLOCK_SIZE_SPATIAL=K,
            GROUP_SIZE=triton.cdiv(N, 32),
            num_warps=8,
            num_stages=2)

    if training:
        grad_scale = torch.empty((input.shape[0], ),
                                  dtype=torch.float32, device="cuda")

        def _update_stats(running_mean, running_var, mean, var, m, n, momentum):
            _new_mean = (running_mean * m + mean * n) / (m + n)
            _var = var + (running_var + ((mean - running_mean) ** 2) * (m / (m + n))) * n
            return _new_mean, _var

        torch._inductor.norm.update_stats(
            running_mean, running_var, mean, var, M, N, momentum)
    else:
        mean.copy_(running_mean)
        var.copy_(running_var)

    R, C = input.shape[-2], input.shape[-1]
    oH, oW = triton.next_power_of_2(R), triton.next_power_of_2(C)
    oRs, oCs = triton.calculate_settings_2d(oH, oW)

    def _contiguous(t):
        if t.stride()[-3:] != (1, R, C):
            return t.contiguous()
        else:
            return t

    running_mean.copy_(mean)
    running_var.copy_(var)

    return post_bn2
