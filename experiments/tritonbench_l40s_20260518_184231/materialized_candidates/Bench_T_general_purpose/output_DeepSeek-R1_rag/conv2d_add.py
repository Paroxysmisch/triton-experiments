import torch
import triton
import triton.language as tl

def _pair(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (x[0], x[0])
        else:
            return tuple(x)
    else:
        return (x, x)

@triton.jit
def conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    B, IC, IH, IW,
    OC, groups, KH, KW,
    OH, OW,
    sH, sW,
    padH, padW,
    dilH, dilW,
    alpha,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    num_blocks_ow = (OW + BLOCK_W - 1) // BLOCK_W
    block_oy = pid_spatial // num_blocks_ow
    block_ox = pid_spatial % num_blocks_ow
    
    oy_start = block_oy * BLOCK_H
    ox_start = block_ox * BLOCK_W
    
    oy = oy_start + tl.arange(0, BLOCK_H)
    ox = ox_start + tl.arange(0, BLOCK_W)
    
    oy_mask = oy < OH
    ox_mask = ox < OW
    mask = oy_mask[:, None] & ox_mask[None, :]
    
    group_size = OC // groups
    group_id = pid_oc // group_size
    ic_start = group_id * (IC // groups)
    ic_end = ic_start + (IC // groups)
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for ky in range(KH):
        for kw in range(KW):
            iy = (oy * sH - padH) + ky * dilH
            ix = (ox * sW - padW) + kw * dilW
            
            iy_valid = (iy >= 0) & (iy < IH)
            ix_valid = (ix >= 0) & (ix < IW)
            i_valid = iy_valid[:, None] & ix_valid[None, :] & mask
            
            for ic in range(ic_start, ic_end):
                input_idx = pid_b * IC * IH * IW + ic * IH * IW + iy * IW + ix
                input_val = tl.load(input_ptr + input_idx, mask=i_valid, other=0.0)
                
                weight_idx = pid_oc * (IC // groups) * KH * KW + (ic - ic_start) * KH * KW + ky * KW + kw
                weight_val = tl.load(weight_ptr + weight_idx)
                
                acc += input_val * weight_val
    
    if bias_ptr != 0:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val
    
    if other_ptr != 0:
        other_idx = pid_b * OC * OH * OW + pid_oc * OH * OW + oy * OW + ox
        other_val = tl.load(other_ptr + other_idx, mask=mask, other=0.0)
        acc += alpha * other_val
    
    output_idx = pid_b * OC * OH * OW + pid_oc * OH * OW + oy * OW + ox
    tl.store(output_ptr + output_idx, acc, mask=mask)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA device"
    device = input.device
    
    B, IC, IH, IW = input.shape
    OC, IC_per_group, KH, KW = weight.shape
    
    assert IC % groups == 0 and OC % groups == 0, "groups must divide in_channels and out_channels"
    assert IC_per_group == IC // groups, "weight's in_channels must be in_channels // groups"
    
    sH, sW = _pair(stride)
    dilH, dilW = _pair(dilation)
    
    if isinstance(padding, str):
        if padding.lower() == 'same':
            padH = ((sH - 1) * IH + dilH * (KH - 1)) // 2
            padW = ((sW - 1) * IW + dilW * (KW - 1)) // 2
        elif padding.lower() == 'valid':
            padH, padW = 0, 0
        else:
            raise ValueError("padding must be 'same', 'valid', or a tuple/int")
    else:
        padH, padW = _pair(padding)
    
    OH = (IH + 2 * padH - dilH * (KH - 1) - 1) // sH + 1
    OW = (IW + 2 * padW - dilW * (KW - 1) - 1) // sW + 1
    
    if out is None:
        out = torch.empty((B, OC, OH, OW), device=device, dtype=input.dtype)
    else:
        assert out.shape == (B, OC, OH, OW), "Output tensor has incorrect shape"
    
    if other is not None:
        if isinstance(other, (int, float)):
            other = torch.full((B, OC, OH, OW), other, device=device, dtype=input.dtype)
        else:
            assert other.shape == (B, OC, OH, OW) or other.numel() == 1, "Other must be broadcastable"
            other = other.expand_as(out) if other.numel() == 1 else other
    else:
        other = None
    
    BLOCK_H, BLOCK_W = 16, 16
    num_blocks_oh = (OH + BLOCK_H - 1) // BLOCK_H
    num_blocks_ow = (OW + BLOCK_W - 1) // BLOCK_W
    num_spatial_blocks = num_blocks_oh * num_blocks_ow
    grid = (B, OC, num_spatial_blocks)
    
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else 0
    other_ptr = other.data_ptr() if other is not None else 0
    output_ptr = out.data_ptr()
    
    conv2d_add_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
        B, IC, IH, IW,
        OC, groups, KH, KW,
        OH, OW,
        sH, sW,
        padH, padW,
        dilH, dilW,
        alpha,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    
    return out
