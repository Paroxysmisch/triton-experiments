import torch
import triton
import triton.language as tl

@triton.jit
def fused_transformer_kernel(
    input_ptr, weight1_ptr, weight2_ptr, residual_ptr, output_ptr, N, D_in, D_k, D_out, stride_input_batch, stride_input_n,
    stride_input_d, stride_weight1_out, stride_weight1_in, stride_weight2_out, stride_weight2_in, stride_res_batch,
    stride_res_n, stride_res_d, stride_output_batch, stride_output_n, stride_output_d, dropout_p, eps, BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    input_ptrs = input_ptr + pid_batch * stride_input_batch + pid_n * stride_input_n
    weight1_ptrs = weight1_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_weight1_out + tl.arange(0, BLOCK_SIZE)[None, :] * stride_weight1_in
    weight2_ptrs = weight2_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_weight2_out + tl.arange(0, BLOCK_SIZE)[None, :] * stride_weight2_in
    res_ptrs = residual_ptr + pid_batch * stride_res_batch + pid_n * stride_res_n
    output_ptrs = output_ptr + pid_batch * stride_output_batch + pid_n * stride_output_n

    offs_d = tl.arange(0, BLOCK_SIZE)
    input_ptrs += offs_d * stride_input_d
    weight1_ptrs += offs_d[None, :] * stride_weight1_in
    weight2_ptrs += offs_d[:, None] * stride_weight2_out
    res_ptrs += offs_d * stride_res_d
    output_ptrs += offs_d * stride_output_d

    input = tl.load(input_ptrs, mask=offs_d < D_in, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptrs, mask=(offs_d[None, :] < D_k) & (offs_d[:, None] < D_in), other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptrs, mask=(offs_d[:, None] < D_out) & (offs_d[None, :] < D_k), other=0.0).to(tl.float32)
    res = tl.load(res_ptrs, mask=offs_d < D_out, other=0.0).to(tl.float32)

    z1 = tl.dot(input, weight1, allow_tf32=False)
    z2 = tl.softmax(z1)
    if tl.rand() < dropout_p:
        z2 = 0.0
    z3 = z2
    z4 = tl.dot(z3, weight2, allow_tf32=False)
    z4 += res
    output = tl.where(offs_d < D_out, z4, 0.0)
    tl.store(output_ptrs, output, mask=offs_d < D_out)
    return

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    check(
        input.shape[-1] == weight1.shape[1],
        f"Incompatible hidden size {input.shape[-1]} and {weight1.shape[1]}",
    )
    check(
        weight1.shape[0] == weight2.shape[1],
        f"Incompatible intermediate size {weight1.shape[0]} and {weight2.shape[1]}",
    )
    check(
        weight2.shape[0] == residual.shape[-1],
        f"Incompatible hidden size {weight2.shape[0]} and last dimension of residual {residual.shape[-1]}",
    )
    has_batch_dim = len(input.shape) > 2
    if has_batch_dim:
        input, batch_dim, N, D_in = pack_input(input)
        residual, _, _, D_out = pack_input(residual)
    else:
        N, D_in = input.shape
        _, _, D_out = residual.shape
    output = torch.empty((N, D_out), device=input.device, dtype=input.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    BLOCK_SIZE = triton.next_power_of_2(D_out)
    with torch.cuda.device(input.device):
        fused_transformer_kernel[grid](
            input,
            weight1,
            weight2,
            residual,
            output,
            N,
            D_in,
            weight1.shape[0],
            weight2.shape[0],
            input.stride(0),
            input.stride(1) if has_batch_dim else 1,
            input.stride(2) if not has_batch_dim else 1,
            weight1.stride(0),
            weight1.stride(1),
            weight2.stride(0),
            weight2.stride(1),
            residual.stride(0),
            residual.stride(1) if has_batch_dim else 1,
            residual.stride(2) if not has_batch_dim else 1,
            output.stride(0),
            output.stride(1) if has_batch_dim else 1,
            output.stride(2) if not has_batch_dim else 1,
            dropout_p,
            eps,
            BLOCK_SIZE,
        )
    if not has_batch_dim:
        output = output.reshape(N, D_out)
    return output
