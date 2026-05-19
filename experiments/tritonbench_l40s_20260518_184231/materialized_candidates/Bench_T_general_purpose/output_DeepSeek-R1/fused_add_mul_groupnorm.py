import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    input1_ptr, input2_ptr, weight_ptr, bias_ptr, output_ptr,
    n_batch, n_channels, h, w, num_groups, eps,
    stride_in1_batch, stride_in1_chan, stride_in1_h, stride_in1_w,
    stride_in2_batch, stride_in2_chan, stride_in2_h, stride_in2_w,
    stride_weight, stride_bias,
    stride_out_batch, stride_out_chan, stride_out_h, stride_out_w,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    num_channels_per_group = n_channels // num_groups
    c_start = pid_group * num_channels_per_group
    
    off_c = tl.arange(0, BLOCK_SIZE_C) + c_start
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_w = tl.arange(0, BLOCK_SIZE_W)
    
    c_mask = (off_c < (pid_group + 1) * num_channels_per_group)
    
    sum_m = tl.zeros((1,), dtype=tl.float32)
    sum_m_sq = tl.zeros((1,), dtype=tl.float32)
    
    for h_idx in range(0, h, BLOCK_SIZE_H):
        for w_idx in range(0, w, BLOCK_SIZE_W):
            h_mask = (h_idx + off_h) < h
            w_mask = (w_idx + off_w) < w
            hw_mask = h_mask[:, None] & w_mask[None, :]
            
            in1_offset = (pid_batch * stride_in1_batch) + (off_c[:, None, None] * stride_in1_chan) + ((h_idx + off_h)[None, :, None] * stride_in1_h) + ((w_idx + off_w)[None, None, :] * stride_in1_w)
            in2_offset = (pid_batch * stride_in2_batch) + (off_c[:, None, None] * stride_in2_chan) + ((h_idx + off_h)[None, :, None] * stride_in2_h) + ((w_idx + off_w)[None, None, :] * stride_in2_w)
            
            mask = c_mask[:, None, None] & hw_mask[None, :, :]
            
            x = tl.load(input1_ptr + in1_offset, mask=mask, other=0.0)
            y = tl.load(input2_ptr + in2_offset, mask=mask, other=0.0)
            
            m = (x + y) * y
            sum_m += tl.sum(m, axis=(0, 1, 2))
            sum_m_sq += tl.sum(m * m, axis=(0, 1, 2))
    
    group_size = num_channels_per_group * h * w
    mean = sum_m / group_size
    var = (sum_m_sq / group_size) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for h_idx in range(0, h, BLOCK_SIZE_H):
        for w_idx in range(0, w, BLOCK_SIZE_W):
            h_mask = (h_idx + off_h) < h
            w_mask = (w_idx + off_w) < w
            hw_mask = h_mask[:, None] & w_mask[None, :]
            
            in1_offset = (pid_batch * stride_in1_batch) + (off_c[:, None, None] * stride_in1_chan) + ((h_idx + off_h)[None, :, None] * stride_in1_h) + ((w_idx + off_w)[None, None, :] * stride_in1_w)
            in2_offset = (pid_batch * stride_in2_batch) + (off_c[:, None, None] * stride_in2_chan) + ((h_idx + off_h)[None, :, None] * stride_in2_h) + ((w_idx + off_w)[None, None, :] * stride_in2_w)
            
            mask = c_mask[:, None, None] & hw_mask[None, :, :]
            
            x = tl.load(input1_ptr + in1_offset, mask=mask, other=0.0)
            y = tl.load(input2_ptr + in2_offset, mask=mask, other=0.0)
            
            m = (x + y) * y
            normalized = (m - mean) * inv_std
            
            gamma = tl.load(weight_ptr + off_c * stride_weight, mask=c_mask, other=0.0)
            beta = tl.load(bias_ptr + off_c * stride_bias, mask=c_mask, other=0.0)
            
            out = normalized * gamma[:, None, None] + beta[:, None, None]
            
            out_offset = (pid_batch * stride_out_batch) + (off_c[:, None, None] * stride_out_chan) + ((h_idx + off_h)[None, :, None] * stride_out_h) + ((w_idx + off_w)[None, None, :] * stride_out_w)
            tl.store(output_ptr + out_offset, out, mask=mask)

class FusedAddMulGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, weight, bias, num_groups, eps):
        assert input1.shape == input2.shape, "input1 and input2 must have the same shape after broadcasting"
        N, C, H, W = input1.shape
        assert C % num_groups == 0, "num_groups must divide number of channels"
        
        output = torch.empty_like(input1)
        
        grid = (N, num_groups)
        BLOCK_SIZE_C = 16
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        
        fused_add_mul_groupnorm_kernel[grid](
            input1, input2, weight, bias, output,
            N, C, H, W, num_groups, eps,
            input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3),
            input2.stride(0), input2.stride(1), input2.stride(2), input2.stride(3),
            weight.stride(0), bias.stride(0),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        ctx.save_for_backward(input1, input2, weight, bias)
        ctx.num_groups = num_groups
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weight, bias = ctx.saved_tensors
        num_groups = ctx.num_groups
        eps = ctx.eps
        
        grad_input1 = grad_input2 = grad_weight = grad_bias = None
        
        with torch.enable_grad():
            input1_ = input1.detach().requires_grad_(True)
            input2_ = input2.detach().requires_grad_(True)
            weight_ = weight.detach().requires_grad_(True)
            bias_ = bias.detach().requires_grad_(True)
            
            output = FusedAddMulGroupNorm.apply(input1_, input2_, weight_, bias_, num_groups, eps)
            grad_input1, grad_input2, grad_weight, grad_bias = torch.autograd.grad(
                outputs=output,
                inputs=(input1_, input2_, weight_, bias_),
                grad_outputs=grad_output,
                retain_graph=True
            )
        
        return grad_input1, grad_input2, grad_weight, grad_bias, None, None, None

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    input2 = input2.expand_as(input1)
    result = FusedAddMulGroupNorm.apply(input1, input2, weight, bias, num_groups, eps)
    if out is not None:
        out.copy_(result)
        return out
    return result
