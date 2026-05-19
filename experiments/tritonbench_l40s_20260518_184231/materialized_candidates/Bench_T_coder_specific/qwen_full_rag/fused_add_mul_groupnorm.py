import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_mul_groupnorm(
    x_ptr, y_ptr, gamma_ptr, beta_ptr, out_ptr, x_reversed_ptr, y_reversed_ptr, 
    B, H, T, C, NHW, stride_x_batch, stride_x_head, stride_y_batch, stride_y_head, 
    stride_z_batch, stride_z_head, stride_x_reversed_batch, stride_x_reversed_head, 
    stride_y_reversed_batch, stride_y_reversed_head, stride_z_reversed_batch, 
    stride_z_reversed_head, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr, GROUP_SIZE_M: tl.constexpr, EPS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(H, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(C, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    x_ptr += pid * NHW
    y_ptr += pid * NHW
    out_ptr += pid * NHW
    x_reversed_ptr += pid * NHW
    y_reversed_ptr += pid * NHW
    gamma_ptr += pid * C
    beta_ptr += pid * C
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(H, C), strides=(stride_x_head, 1), offsets=(
        pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    y_block_ptr = tl.make_block_ptr(base=y_ptr, shape=(H, C), strides=(stride_y_head, 1), offsets=(
        pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    out_block_ptr = tl.make_block_ptr(base=out_ptr, shape=(H, C), strides=(stride_z_head, 1), offsets=(
        pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    x_reversed_block_ptr = tl.make_block_ptr(base=x_reversed_ptr, shape=(H, C), strides=(
        stride_x_reversed_head, 1), offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    y_reversed_block_ptr = tl.make_block_ptr(base=y_reversed_ptr, shape=(H, C), strides=(
        stride_y_reversed_head, 1), offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    x = tl.load(x_block_ptr)
    y = tl.load(y_block_ptr)
    x_reversed = tl.load(x_reversed_block_ptr)
    y_reversed = tl.load(y_reversed_block_ptr)
    xmy = tl.where(x >= x_reversed, x - y_reversed, x - y)
    ymy = tl.where(y >= y_reversed, y - x_reversed, y - x)
    max_xy = tl.where(x >= y, xmy, ymy)
    rstd = 1 / tl.sqrt(max_xy + EPS)
    gamma = tl.load(gamma_ptr)[None, :]
    beta = tl.load(beta_ptr)[None, :]
    z = (x - y) * rstd * gamma + beta
    tl.store(out_block_ptr, z)


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, out=None):
    f = torch.tensor([6.90775527898214e-08, 1.38155105579643e-07, 2.76310211159286e-07, 5.52620422318571e-07,
                      1.10524084463714e-06, 2.21048168927428e-06, 4.42096337854857e-06, 8.84192675709714e-06,
                      1.76838535141943e-05, 3.53677070283885e-05, 7.07354140567771e-05, 0.000141470828113543,
                      0.000282941656227085, 0.00056588331245417, 0.00113176662490834, 0.00226353324981668,
                      0.00452706649963336, 0.00905413299926672, 0.0181082659985334, 0.0362165319970668,
                      0.0724330639941337, 0.144866127988267, 0.289732255976534, 0.579464511953068, 1.15892902390614,
                      2.31785804781228, 4.63571609562456, 9.27143219124912, 18.5428643824982, 37.0857287649964,
                      74.1714575299928, 148.342915059986, 296.685830119972, 593.371660239944, 1186.74332047989,
                      2373.48664095978, 4746.97328191956, 9493.94656383911, 18987.8931276782, 37975.7862553564,
                      75951.5725107128, 151903.145021425, 303806.29004285, 607612.5800857, 1215225.1601714,
                      2430450.3203428])
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    if out is None:
        out = torch.empty_like(input1)
    else:
        assert out.shape == input1.shape and out.is_contiguous()
    B, H, T, C = input1.shape
    NHW = T * H
    MAX_FUSED_SIZE = 65536 // input1.element_size()
    BLOCK_SIZE_M = min(MAX_FUSED_SIZE, triton.next_power_of_2(T))
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    grid = (triton.cdiv(NHW, BLOCK_SIZE_M) * triton.cdiv(C, BLOCK_SIZE_N), B * H)
    num_warps = 4
    x_reversed = input1.clone().flip(-1)
    y_reversed = input2.clone().flip(-1)
    _fused_add_mul_groupnorm[grid](input1, input2, weight, bias, out, x_reversed, y_reversed, B, H
