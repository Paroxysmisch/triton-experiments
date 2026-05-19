import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(q_tile_ptr, k_tile_ptr, v_tile_ptr, O_tile_ptr, 
                scaling_factor_ptr, mask_ptr, 
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk, 
                stride_oz, stride_oh, stride_om, stride_on, 
                H, N_Q, DMODEL): 
    ...
    # TODO: Implement the dot-product attention calculation and write the result to O_tile_ptr

@triton.jit
def _bwd_preprocess(DO_tile_ptr, LO_tile_ptr, DK_tile_ptr, DV_tile_ptr, 
                    scaling_factor_ptr, LO, 
                    BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
                    stride_qz, stride_qh, stride_qm, stride_qk,
                    stride_kz, stride_kh, stride_kn, stride_kk, 
                    stride_oz, stride_oh, stride_om, stride_on, 
                    H, N_Q, DMODEL):
    ...
    # TODO: Implement the backward pass preprocessing part and write results to DK, DV

@triton.jit
def _bwd_kernel(Q, K, V, DQ, DK, DV, Q_scaling_factors, K_scaling_factors,
                attn_mask, 
                O, DO, 
                grid_Q, grid_K, grid_V, grid_O, 
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk, 
                stride_oz, stride_oh, stride_om, stride_on, 
                H, N_Q, DMODEL, 
                use_custom_loader=False, device_id: tl.constexpr=0):
    ...
    # TODO: Implement the backward pass kernel part and write results back to DQ, DK, DV

class attention:

    def __init__(self, device_id=0, BLOCK=128, num_warps=8, num_stages=3, use_custom_loader=False):
        self.device_id = device_id
        self.use_custom_loader = use_custom_loader
        self.BLOCK_M = BLOCK
        self.BLOCK_N = BLOCK
        self.BLOCK_DMODEL = BLOCK
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, query, key, value, Q_scaling_factors, K_scaling_factors, mask):
        ...
        # TODO: Call the wrapped method & return the output

    def backward(self, query, key, value, output, d_output, Q_scaling_factors, K_scaling_factors, mask):
        ...
        # TODO: Call the wrapped method to compute the gradients & return d_query, d_key, d_value
