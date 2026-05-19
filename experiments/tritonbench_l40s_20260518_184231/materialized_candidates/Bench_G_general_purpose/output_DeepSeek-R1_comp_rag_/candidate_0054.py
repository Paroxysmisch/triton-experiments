import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_oc,
    weight_stride_ic,
    weight_stride_kh,
    weight_stride_kw,
    output_stride_n,
    output_stride_oc,
    output_stride_oh,
    output_stride_ow,
    N,
    C,
    H,
    W,
    OC,
    KH,
    KW,
    OH,
    OW,
    PH,
    PW,
    SH,
    SW,
    groups,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNELS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    num_ow_blocks = (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    block_oh = pid_spatial // num_ow_blocks
    block_ow = pid_spatial % num_ow_blocks
    
    batch_start = pid_batch * BLOCK_SIZE_BATCH
    oc_start = pid_oc * BLOCK_SIZE_OUT_CHANNELS
    oh_start = block_oh * BLOCK_SIZE_H
    ow_start = block_ow * BLOCK_SIZE_W
    
    batch_offset = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    oc_offset = oc_start + tl.arange(0, BLOCK_SIZE_OUT_CHANNELS)
    oh_offset = oh_start + tl.arange(0, BLOCK_SIZE_H)
    ow_offset = ow_start + tl.arange(0, BLOCK_SIZE_W)
    
    group = oc_start // (OC // groups)
    ic_per_group = C // groups
    ic_start = group * ic_per_group
    
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNELS, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=ACC_TYPE)
    
    for kh in range(KH):
        for kw in range(KW):
            for ic_block in range(ic_start, ic_start + ic_per_group):
                ic = ic_block
                input_h = oh_offset[:, None] * SH + kh - PH
                input_w = ow_offset[None, :] * SW + kw - PW
                
                valid_h = (input_h >= 0) & (input_h < H)
                valid_w = (input_w >= 0) & (input_w < W)
                valid = valid_h & valid_w
                
                input_n = tl.max_contiguous(tl.where(batch_offset < N, batch_offset, 0), BLOCK_SIZE_BATCH)
                input_c = ic
                
                input_val = tl.load(
                    input_ptr + input_n[:, None, None, None] * input_stride_n +
                    input_c * input_stride_c +
                    input_h[:, None] * input_stride_h +
                    input_w[None, :] * input_stride_w,
                    mask=valid[None, None, :, :] & (input_n[:, None, None, None] < N),
                    other=0.0
                )
                
                weight_val = tl.load(
                    weight_ptr + oc_offset[:, None, None, None] * weight_stride_oc +
                    (ic - ic_start) * weight_stride_ic +
                    kh * weight_stride_kh +
                    kw * weight_stride_kw,
                    mask=oc_offset[:, None, None, None] < OC,
                    other=0.0
                )
                
                acc += tl.sum(input_val * weight_val, axis=1)
    
    output_n = batch_offset[:, None, None, None]
    output_oc = oc_offset[None, :, None, None]
    output_oh = oh_offset[None, None, :, None]
    output_ow = ow_offset[None, None, None, :]
    
    output_idx = (
        output_n * output_stride_n +
        output_oc * output_stride_oc +
        output_oh * output_stride_oh +
        output_ow * output_stride_ow
    )
    
    tl.store(
        output_ptr + output_idx,
        acc,
        mask=(
            (output_n < N) &
            (output_oc < OC) &
            (output_oh < OH) &
            (output_ow < OW)
        )
    )

def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    groups: int = 1,
    fp16: bool = False,
    tf32: bool = False
) -> torch.Tensor:
    N, C, H, W = input.shape
    OC, IC, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    
    output = torch.empty((N, OC, OH, OW), device=input.device, dtype=input.dtype)
    
    BLOCK_SIZE_BATCH = 1
    BLOCK_SIZE_OUT_CHANNELS = 32
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid_batch = (N + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
    grid_oc = (OC + BLOCK_SIZE_OUT_CHANNELS - 1) // BLOCK_SIZE_OUT_CHANNELS
    grid_oh = (OH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_ow = (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (grid_batch, grid_oc, grid_oh * grid_ow)
    
    acc_type = tl.float32
    if fp16:
        acc_type = tl.float16
    
    conv2d_forward_kernel[grid](
        input,
        weight,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        N,
        C,
        H,
        W,
        OC,
        KH,
        KW,
        OH,
        OW,
        PH,
        PW,
        SH,
        SW,
        groups,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_OUT_CHANNELS=BLOCK_SIZE_OUT_CHANNELS,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        ACC_TYPE=acc_type,
        ALLOW_TF32=tf32,
    )
    
    return output
