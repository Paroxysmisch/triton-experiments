import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


@triton.jit
def _grid_sampler_2d_bilinear(
    img_ptr,
    x0,
    y0,
    tx,
    ty,
    rcp_h,
    rcp_w,
    batch,
    c,
    h,
    w,
    img_stride_batch,
    img_stride_c,
    img_stride_h,
    img_stride_w,
    A,
    out_ptr,
    indices_ptr,
    padding_mode,
    index_scale,
    index_offset,
    n_samples,
    n_channels,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    EVEN_C: tl.constexpr,
    BILINEAR_INTERP: tl.constexpr,
):
    pid = tl.program_id(0)
    out_img_offs = tl.arange(0, BLOCK_SIZE_H)[:, None] * n_channels + tl.arange(
        0, BLOCK_SIZE_C
    )[None, :]
    out_img_mask = (pid < n_samples)[:, None] & (out_img_offs < n_channels)[None, :]
    out_img_ptrs = out_ptr + pid * n_channels + out_img_offs
    out_vals = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_C), dtype=tl.float32)

    if BILINEAR_INTERP:
        vtop = tl.load(img_ptr + tl.minimum(h - 1, y0.to(tl.int64)), eviction_policy="evict_last")
        vbottom = tl.load(img_ptr + tl.minimum(h - 1, (y0 + 1).to(tl.int64)), eviction_policy="evict_last")

        vleft_top = tl.load(
            img_ptr + tl.minimum(w - 1, x0.to(tl.int64)),
            eviction_policy="evict_last",
        )
        vright_top = tl.load(
            img_ptr + tl.minimum(w - 1, (x0 + 1).to(tl.int64)),
            eviction_policy="evict_last",
        )

        vleft_bottom = tl.load(
            img_ptr + tl.minimum(w - 1, (x0).to(tl.int64)),
            eviction_policy="evict_last",
        )
        vright_bottom = tl.load(
            img_ptr + tl.minimum(w - 1, (x0 + 1).to(tl.int64)),
            eviction_policy="evict_last",
        )

        out_vals += (
            (vtop * (1 - tx) + vbottom * tx) * (1 - ty) + (vleft_top * (1 - tx) + vleft_bottom * tx) * ty
        )
    else:
        val = tl.load(
            img_ptr + tl.minimum(h - 1, y0.to(tl.int64)) * w + tl.minimum(w - 1, x0.to(tl.int64)),
            eviction_policy="evict_last",
        )
        out_vals += val

    if padding_mode == "zeros":
        out_vals = tl.where(x0 >= 0, out_vals, 0)
        out_vals = tl.where(x0 < w, out_vals, 0)
        out_vals = tl.where(y0 >= 0, out_vals, 0)
        out_vals = tl.where(y0 < h, out_vals, 0)
    elif padding_mode == "border":
        x0 = tl.where(x0 < 0, 0, x0)
        x0 = tl.where(x0 > w - 1, w - 1, x0)
        y0 = tl.where(y0 < 0, 0, y0)
        y0 = tl.where(y0 > h - 1, h - 1, y0)
        out_vals = tl.where(out_img_mask, out_vals, 0)
    else:  # "reflection"
        abs_x0 = tl.abs(x0)
        abs_y0 = tl.abs(y0)
        floor_x0 = tl.floor(abs_x0)
        ceil_x0 = floor_x0 + 1
        dist_to_floor = abs_x0 - floor_x0
        dist_to_ceil = 1 - dist_to_floor
        floor_x0 = floor_x0.to(tl.int64)
        ceil_x0 = ceil_x0.to(tl.int64)
        reflected_x0 = -abs_x0
        sign_x0 = tl.where(x0 >= 0, 1.0, -1.0)
        refld_floor_x0 = tl.floor(reflected_x0)
        refld_ceil_x0 = refld_floor_x0 + 1
        refl_dist_to_floor = reflected_x0 - refld_floor_x0
        refl_dist_to_ceil = 1 - refl_dist_to_floor
        refld_floor_x0 = refld_floor_x0.to(tl.int64)
        refld_ceil_x0 = refld_ceil_x0.to(tl.int64)

        v1 = tl.load(
            img_ptr + (h - abs_y0 - 1) * w + (floor_x0),
            mask=out_img_mask & (dist_to_floor <= ((w + 1) / 2)),
            eviction_policy="evict_last",
        ) * dist_to_floor
        v2 = tl.load(
            img_ptr + (h - abs_y0 - 1) * w + (refld_floor_x0),
            mask=out_img_mask & (refl_dist_to_floor <= ((w + 1) / 2)),
            eviction_policy="evict_last",
        ) * refl_dist_to_floor
        out_vals += (v1 + v2) * sign_x0

    if indices_ptr is not None:
        idx = (
            ((pid + index_offset) * n_channels + out_img_offs) // BLOCK_SIZE_C
        )
        tl.store(indices_ptr + idx, out_vals.sum(axis=0))
    
    tl.store(out_img_ptrs, out_vals, mask=out_img_mask)


@triton.jit
def _grid_sampler_nd_bilinear(
    img_ptr,
    idx,
    base,
    strides,
    offsets,
    scale,
    inverse_scale,
    padding_mode,
    index_scale,
    index_offset,
    ndim_img,
    n_elems,
    n_indices_per_elem,
    idx_stride_n_elems,
    idx_stride_n_indices_per_elem,
    int_dtype_code,
    float_dtype_code,
    BLOCK_N_INDICES: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    stride_scale = triton_helpers.stride_of(strides, ndim_img)
    stride_inverse_scale = triton_helpers.stride_of(strides, ndim_img + 1)
    stride_padding_mode = triton_helpers.stride_of(strides, ndim_img + 2)

    m = tl.program_id(0)
    img_m = tl.load(img_ptr + m)
    idx_n = tl.arange(0, BLOCK_N_INDICES) // n_indices_per_elem
    idx_m = tl.arange(0, BLOCK_N_INDICES) % n_indices_per_elem
    mask_idx = idx_n[:, None] < n_elems[None, :]
    mask_img = True
    if img_ptr.dtype.is_int64():
        if int_dtype_code == 1:
            img_base = tl.load(base + tl.load(img_ptr + m))
        elif int_dtype_code == 2:
            img_strides = tl.load(strides + tl.load(img_ptr + m) * stride_scale)
            img_base = tl.load(img_strides)
        elif int_dtype_code == 3:
            img_strides = tl.load(
                strides + tl.load(img_ptr + m) * stride_scale, eviction_policy="evict_last"
            )
            img_inverse_scale = tl.load(
                img_strides + stride_inverse_scale, eviction_policy="evict_last"
            )
            img_base = tl.load(img_strides)
            img_inverse_scale = img_inverse_scale.to(idx_n.dtype)
            img_scale = tl.load(scale + tl.load(img_ptr + m)).to(idx_n.dtype)
            img_padding_mode = tl.load(
                img_strides + stride_padding_mode, eviction_policy="evict_last"
            ).to(tl.int64)
    else:
        img_val = tl.load(img_ptr + m).to(tl.int64)
        if int_dtype_code == 1:
            img_base = tl.load(base + img_val)
        elif int_dtype_code == 2:
            img_strides = tl.load(strides + img_val * stride_scale)
            img_base = tl.load(img_strides)
        elif int_dtype_code == 3:
            img_strides = tl.load(strides + img_val * stride_scale, eviction_policy="evict_last")
            img_inverse_scale = tl.load(
                img_strides + stride_inverse_scale, eviction_policy="evict_last"
            )
            img_base = tl.load(img_strides)
            img_inverse_scale = img_inverse_scale.to(idx_n.dtype)
            img_scale = tl.load(scale + img_val).to(idx_n.dtype)
            img_padding_mode = tl.load(
                img_strides + stride_padding_mode, eviction_policy="evict_last"
            ).to(tl.int64)
    idx_block = tl.load(
        idx + idx_n[:, None] * idx_stride_n_elems + idx_m[None, :] * idx_stride_n_indices_per_elem,
        mask=mask_idx,
    )
    idx_float = idx_block.to(tl.float32)
    idx_scaled = idx_float * scale
    if img_ptr.dtype.is_int64():
        if int
