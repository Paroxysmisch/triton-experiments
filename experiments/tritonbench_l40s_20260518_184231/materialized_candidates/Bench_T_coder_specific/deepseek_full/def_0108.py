import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import FunctionCtx
from torch.utils._triton import has_triton

if has_triton():
    @triton.jit
    def _grid_sample_with_affine_kernel(
        input_ptr, output_ptr, theta_ptr,
        N, C, H_in, W_in, H_out, W_out,
        size_mult_ptr,
        padding_mode_is_zeros,
        align_corners_int,
        BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
        GROUP_SIZE_N: tl.constexpr,
        GROUP_SIZE_C: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_W: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_c = tl.cdiv(C, BLOCK_C)
        num_pid_in_group = GROUP_SIZE_N * GROUP_SIZE_C
        group_id = pid // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        first_pid_c = group_id * GROUP_SIZE_C
        pid_n = first_pid_n + (pid % GROUP_SIZE_N)
        pid_c = first_pid_c + (pid % GROUP_SIZE_C)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        h_out = tl.arange(0, BLOCK_H)
        w_out = tl.arange(0, BLOCK_W)
        size_mult = tl.load(size_mult_ptr)
        h_in = (h_out + 0.5) / H_out * H_in - 0.5
        w_in = (w_out + 0.5) / W_out * W_in - 0.5
        if padding_mode_is_zeros:
            h_in = tl.where(h_in < 0, 0., tl.where(h_in >= H_in, H_in - 1., h_in))
            w_in = tl.where(w_in < 0, 0., tl.where(w_in >= W_in, W_in - 1., w_in))
        elif align_corners:
            h_in = tl.where(h_in < 0, 0., tl.where(h_in >= H_in, H_in - 1., h_in))
            w_in = tl.where(w_in < 0, 0., tl.where(w_in >= W_in, W_in - 1., w_in))
        else:
            h_in = (h_in + 0.5) / size_mult - 0.5
            w_in = (w_in + 0.5) / size_mult - 0.5
        h_in_floats = h_in + tl.arange(0, BLOCK_H * BLOCK_W * BLOCK_N * BLOCK_C).to(tl.float32) % (H_in - 1)
        w_in_floats = w_in + tl.arange(0, BLOCK_H * BLOCK_W * BLOCK_N * BLOCK_C).to(tl.float32) % (W_in - 1)
        theta_ptr_n = theta_ptr + n[:, None] * 2
        theta_ptr_c = theta_ptr_n + c[None, :] * 3
        h_in_scaled = h_in_floats * 0.5
        w_in_scaled = w_in_floats * 0.5
        h_in_floats = h_in_floats * 2
        w_in_floats = w_in_floats * 2
        affine_h = tl.load(theta_ptr_n)
        affine_w = tl.load(theta_ptr_n + 1)
        affine_h_plus_w = tl.load(theta_ptr_c)
        affine_h_minus_w = tl.load(theta_ptr_c + 1)
        affine_v = tl.load(theta_ptr_c + 2)
        h_sample = (
            affine_h * h_in_scaled + affine_w * w_in_scaled + affine_h_plus_w * h_in_floats + affine_h_minus_w * w_in_floats + affine_v
        )
        w_sample = (
            affine_h * w_in_scaled + affine_w * h_in_scaled + affine_h_plus_w * w_in_floats - affine_h_minus_w * h_in_floats + affine_v
        )
        input_ptr_n = input_ptr + n[:, None] * C
        input_ptr_c = input_ptr_n + c[None, :]
        output_ptr_n = output_ptr + n[:, None] * H_out
        output_ptr_h = output_ptr_n + h_out[None, :]
        output_ptr_n_w = output_ptr_h + w_out[None, :] * H_out
        grid_sample_4d(
            input_ptr_c,
            output_ptr_n_w,
            input_ptr,
            C,
            H_in,
            W_in,
            h_sample,
            w_sample,
            padding_mode_is_zeros,
            align_corners_int,
            BLOCK_H,
            BLOCK_W,
            BLOCK_N,
            BLOCK_C,
        )

    def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
        if theta.requires_grad:
            ctx = FunctionCtx()
            grad_sample = torch.empty_like(input)
            grad_theta = torch.empty_like(theta)
            ctx.save(grad_sample, grad_theta)
            torch.cuda.set_device(input.device)
            ctx.set_ctx(input.device)
            grad_sample.fill_(0)
            grad_theta.fill_(0)
            grid_sample_with_affine_backward(input, grad_sample, theta, grad_theta, size, mode, padding_mode, align_corners)
            input = input.add(grad_sample)
        N, C, H_in, W_in = input.shape
        H_out, W_out = size
        num_pid_n = triton.cdiv(N, 32)
        num_pid_c = triton.cdiv(C, 32)
        num_pid = num_pid_n * num_pid_c
        grid_out = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
        padding_mode_is_zeros = padding_mode == 'zeros'
        align_corners_int = int(align_corners)
        if mode == 'bilinear':
            _grid_sample_with_affine_kernel[(num_pid,)](
                input, grid_out, theta,
                N, C, H_in, W_in, H_out, W_out,
                size_mult=torch.tensor(H_in / H_out, device=input.device),
