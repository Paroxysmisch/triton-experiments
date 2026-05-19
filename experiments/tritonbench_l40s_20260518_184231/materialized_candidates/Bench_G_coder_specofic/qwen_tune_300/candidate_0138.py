import torch
import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V,  # query, key, and value tensors
    layout_csr_row_indices, layout_csr_col_indices,  # CSR layout data
    Out,  # output tensor
    stride_qbs, stride_qh, stride_qd,  # strides for input query tensor
    stride_kbs, stride_kh, stride_kd,  # strides for input key tensor
    stride_vbs, stride_vh, stride_vd,  # strides for input value tensor
    stride_obs, stride_oh, stride_od,  # strides for output tensor
    stride_k_cache_bs, stride_k_cache_sh, stride_k_cache_sd,  # strides for k cache
    stride_k_cache_h_bs, stride_k_cache_h_sh, stride_k_cache_h_sd,  # strides for k cache
    stride_k_cache_h2_bs, stride_k_cache_h2_sh, stride_k_cache_h2_sd,  # strides for k cache
    stride_k_cache_h3_bs, stride_k_cache_h3_sh, stride_k_cache_h3_sd,  # strides for k cache
    stride_k_cache_h4_bs, stride_k_cache_h4_sh, stride_k_cache_h4_sd,  # strides for k cache
    stride_k_cache_h5_bs, stride_k_cache_h5_sh, stride_k_cache_h5_sd,  # strides for k cache
    stride_k_cache_h6_bs, stride_k_cache_h6_sh, stride_k_cache_h6_sd,  # strides for k cache
    stride_k_cache_h7_bs, stride_k_cache_h7_sh, stride_k_cache_h7_sd,  # strides for k cache
    stride_k_cache_h8_bs, stride_k_cache_h8_sh, stride_k_cache_h8_sd,  # strides for k cache
    stride_k_cache_h9_bs, stride_k_cache_h9_sh, stride_k_cache_h9_sd,  # strides for k cache
    stride_k_cache_h10_bs, stride_k_cache_h10_sh, stride_k_cache_h10_sd,  # strides for k cache
    stride_k_cache_h11_bs, stride_k_cache_h11_sh, stride_k_cache_h11_sd,  # strides for k cache
    stride_k_cache_h12_bs, stride_k_cache_h12_sh, stride_k_cache_h12_sd,  # strides for k cache
    stride_k_cache_h13_bs, stride_k_cache_h13_sh, stride_k_cache_h13_sd,  # strides for k cache
    stride_k_cache_h14_bs, stride_k_cache_h14_sh, stride_k_cache_h14_sd,  # strides for k cache
    stride_k_cache_h15_bs, stride_k_cache_h15_sh, stride_k_cache_h15_sd,  # strides for k cache
    stride_k_cache_h16_bs, stride_k_cache_h16_sh, stride_k_cache_h16_sd,  # strides for k cache
    stride_k_cache_h17_bs, stride_k_cache_h17_sh, stride_k_cache_h17_sd,  # strides for k cache
    stride_k_cache_h18_bs, stride_k_cache_h18_sh, stride_k_cache_h18_sd,  # strides for k cache
    stride_k_cache_h19_bs, stride_k_cache_h19_sh, stride_k_cache_h19_sd,  # strides for k cache
    stride_k_cache_h20_bs, stride_k_cache_h20_sh, stride_k_cache_h20_sd,  # strides for k cache
    stride_k_cache_h21_bs, stride_k_cache_h21_sh, stride_k_cache_h21_sd,  # strides for k cache
    stride_k_cache_h22_bs, stride_k_cache_h22_sh, stride_k_cache_h22_sd,  # strides for k cache
    stride_k_cache_h23_bs, stride_k_cache_h23_sh, stride_k_cache_h23_sd,  # strides for k cache
    stride_k_cache_h24_bs, stride_k_cache_h24_sh, stride_k_cache_h24_sd,  # strides for k cache
    stride_k_cache_h25_bs, stride_k_cache_h25_sh, stride_k_cache_h25_sd,  # strides for k cache
    stride_k_cache_h26_bs, stride_k_cache_h26_sh, stride_k_cache_h26_sd,  # strides for k cache
    stride_k_cache_h27_bs, stride_k_cache_h27_sh, stride_k_cache_h27_sd,  # strides for k cache
    stride_k_cache_h28_bs, stride_k_cache_h28_sh, stride_k_cache_h28_sd,  # strides for k cache
    stride_k_cache_h29_bs, stride_k_cache_h29_sh, stride_k_cache_h29_sd,  # strides for k cache
    stride_k_cache_h30_bs, stride_k_cache_h30_sh, stride_k_cache_h30_sd,  # strides for k cache
    stride_k_cache_h31_bs, stride_k_cache_h31_sh, stride_k_cache_h31_sd,  # strides for k cache
    stride_k_cache_h32_bs, stride_k_cache_h32_sh, stride_k_cache_h32_sd,  # strides for k cache
    stride_k_cache_h33_bs, stride_k_cache_h33_sh, stride_k_cache_h33_sd,  # strides for k cache
    stride_k_cache_h34_bs, stride_k_cache_h34_sh, stride_k_cache_h34_sd,  # strides for k cache
    stride_k_cache_h35_bs, stride_k_cache_h35_sh, stride_k_cache_h35_sd,  # strides for k cache
    stride_k_cache_h36_bs, stride_k_cache_h36_sh, stride_k_cache_h36_sd,  # strides for k cache
    stride_k_cache_h37_bs, stride_k_cache_h37_sh, stride_k_cache_h37_sd,  # strides for k cache
    stride_k_cache_h38_bs, stride_k_cache_h38_sh, stride_k_cache_h38_sd,  # strides for k cache
    stride_k_cache_h39_bs, stride_k_cache_h39_sh, stride_k_cache_h39_sd,  # strides for k cache
    stride_k_cache_h40_bs, stride_k_cache_h40_sh, stride_k_cache_h40_sd,  # strides for k cache
    stride_k_cache_h41_bs, stride_k_cache_h41_sh, stride_k_cache_h41_sd,  # strides for k cache
    stride_k_cache_h42_bs, stride_k_cache_h42_sh, stride_k_cache_h42_sd,  # strides for k cache
    stride_k_cache_h43_bs, stride_k_cache_h43_sh, stride_k_cache_h43_sd,  # strides for k cache
    stride_k_cache_h44_bs, stride_k_cache_h44_sh, stride_k_cache_h44_sd,  # strides for k cache
    stride_k_cache_h45_bs, stride_k_cache_h45_sh, stride_k_cache_h45_sd,  # strides for k cache
    stride_k_cache_h46_bs, stride_k_cache_h46_sh, stride_k_cache_h46_sd,  # strides for k cache
    stride_k_cache_h47_bs, stride_k_cache_h47_sh, stride_k_cache_h47_sd,  # strides for k cache
    stride_k_cache_h48_bs, stride_k_cache_h48_sh, stride_k_cache_h48_sd,  # strides for k cache
    stride_k_cache_h49_bs, stride_k_cache_h49_sh, stride_k_cache_h49_sd,  # strides for k cache
    stride_k_cache_h50_bs, stride_k_cache_h50_sh, stride_k_cache_h50_sd,  # strides for k cache
    stride_k_cache_h51_bs, stride_k_cache_h51_sh, stride_k_cache_h51_sd,  # strides for k cache
    stride_k_cache_h52_bs, stride_k_cache_h52_sh, stride_k_cache_h52_sd,  # strides for k cache
    stride_k_cache_h53_bs, stride_k_cache_h53_sh, stride_k_cache_h53_sd,  # strides for k cache
    stride_k_cache_h54_bs, stride_k_cache_h54_sh, stride_k_cache_h54_sd,  # strides for k cache
    stride_k_cache_h55_bs, stride_k_cache_h55_sh, stride_k_cache_h55_sd,  # strides for k cache
    stride_k_cache_h56_bs, stride_k_cache_h56_sh, stride_k_cache_h56_sd,  # strides for k cache
    stride_k_cache_h57_bs, stride_k_cache_h57_sh, stride_k_cache_h57_sd,  # strides for k cache
    stride_k_cache_h58_bs, stride_k_cache_h58_sh, stride_k_cache_h58_sd,  # strides for k cache
    stride_k_cache_h59_bs, stride_k_cache_h59_sh, stride_k_cache_h59_sd,  # strides for k cache
    stride_k_cache_h60_bs, stride_k_cache_h60_sh
