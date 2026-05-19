import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_forward(x_ptr, out_ptr, N, C, H, W, OH, OW, stride_h, stride_w, pad_h, pad_w, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    y_offset = tl.program_id(1) * YBLOCK
    
    x_mask = x_offset < N
    y_mask = y_offset < OH
    
    x_coords = x_offset * stride_h + pad_h + tl.arange(0, XBLOCK)
    y_coords = y_offset * stride_w + pad_w + tl.arange(0, YBLOCK)
    
    x_coords = tl.clip(x_coords, 0, H)
    y_coords = tl.clip(y_coords, 0, W)
    
    acc = tl.zeros([XBLOCK, YBLOCK, C], dtype=tl.float32)
    count = tl.zeros([XBLOCK, YBLOCK, C], dtype=tl.float32)
    
    for c in range(C):
        for h in range(H):
            for w in range(W):
                x_idx = tl.index_select(h, x_coords)
                y_idx = tl.index_select(w, y_coords)
                acc[x_idx, y_idx, c] += tl.load(x_ptr + (h * W + w) * C + c)
                count[x_idx, y_idx, c] += 1
    
    avg_val = acc / count
    tl.store(out_ptr + (y_offset * OW + x_offset) * C + c, avg_val, mask=x_mask & y_mask)
