output,
    batch, nnz, m, n, trans_a, trans_b, beta, acc_dtype, allow_tf32
):
    # Function implementation here

def _run_sampled_addmm_kernel(
    alpha, beta, is_beta_zero, k, values, crow_indices, col_indices,
    mat1, mat2, acc_dtype, allow_tf32
):
    # Function implementation here

def _run_bsr_softmax_kernel(crow_indices, values, row_block, col_block, tile_size):
    # Function implementation here
