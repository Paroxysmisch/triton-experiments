import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False) -> tl.Tensor:
    # Define constants
    BLOCK_SIZE_I = 8
    BLOCK_SIZE_J = 8
    BLOCK_SIZE_K = 8
    
    # Get shapes
    N, C_in, H_in, W_in = input.shape
    C_out, _, kH, kW = weight.shape
    
    # Compute output dimensions
    H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    # Allocate output tensor
    output = tl.zeros((N, C_out, H_out, W_out), dtype=input.dtype)
    
    # Shared memory for intermediate results
    s_output = tl.zeros_like(output, num_bytes=1024)
    
    # Convolution loop
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    
    i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    i = i % H_out
    j = j % W_out
    k_start = k * stride - padding
    
    mask = (i < H_out) & (j < W_out)
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K), dtype=input.dtype)
    
    for l in range(C_in // groups):
        w_l = weight[:, l * groups:(l + 1) * groups, :, :]
        b_l = bias[l * groups:(l + 1) * groups] if bias is not None else 0
        
        for kh in range(kH):
            kh_dilated = kh * dilation
            for kw in range(kW):
                kw_dilated = kw * dilation
                
                idx_i = i + kh_dilated
                idx_j = j + kw_dilated
                
                valid_idx = (idx_i >= 0) & (idx_i < H_in) & (idx_j >= 0) & (idx_j < W_in)
                
                x = tl.load(input + ((pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I))[:, None, None] * H_in * W_in +
                                    (pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J))[None, :, None] * W_in +
                                    idx_i[:, None, None] * W_in + idx_j)[valid_idx],
                            mask=valid_idx, boundary_check=True)
                
                w = w_l[l, :, kh_dilated:kH:dilation, kw_dilated:kW:dilation]
                
                acc += tl.dot(x, w)
        
        acc += b_l[None, :, None]
        
        s_output[i, j, :] = acc
    
    # Batch normalization loop
    s_mean = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=output.dtype)
    s_variance = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=output.dtype)
    
    for l in range(C_out):
        gamma = bn_weight[l] if bn_weight is not None else 1.0
        beta = bn_bias[l] if bn_bias is not None else 0.0
        
        for k in range(H_out):
            for m in range(W_out):
                x = s_output[k, m, l]
                mean = tl.mean(x)
                variance = tl.variance(x)
                
                s_mean[k, m] += mean
                s_variance[k, m] += variance
                
                if training:
                    running_mean[l] = running_mean[l] * (1 - momentum) + mean * momentum
                    running_var[l] = running_var[l] * (1 - momentum) + variance * momentum
                
                x_normalized = (x - mean) / tl.sqrt(variance + eps)
                output[k, l, k, m] = gamma * x_normalized + beta
    
    return output
