def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on):
    # Iterate over context size
    for n_ctx in N_CTX:
        # Initialize pointers and offsets
        Q_ptr = Q + stride_qz * n_ctx + stride_qh * head_id + stride_qm * batch_id
        K_ptr = K + stride_kz * n_ctx + stride_kh * head_id + stride_kn * batch_id
        V_ptr = V + stride_vz * n_ctx + stride_vh * head_id + stride_vk * batch_id
        Out_ptr = Out + stride_oz * n_ctx + stride_oh * head_id + stride_om * batch_id

        # Invoke _attn_fwd_inner for each block
        _attn_fwd_inner(Q_ptr, K_ptr, V_ptr, Q_scale, K_scale, Out_ptr, stride_qk, stride_kk, stride_vn, stride_on)

def _attn_fwd_inner(Q_ptr, K_ptr, V_ptr, Q_scale, K_scale, Out_ptr, stride_qk, stride_kk, stride_vn, stride_on):
    # Load blocks of K and V
    K_blocks = load_blocks(K_ptr, stride_kk, BLOCK_N)
    V_blocks = load_blocks(V_ptr, stride_vn, BLOCK_N)

    # Compute the scaled dot-product qk
    qk = compute_qk(Q_ptr, K_ptr, Q_scale, K_scale, stride_qk)

    # Apply the softmax operation
    p = softmax(qk)

    # Accumulate the result of p weighted by V
    acc = accumulate(p, V_blocks)

    # Update normalization l_i and maximum score m_i
    l_i, m_i = update_normalization(acc)

    # Store the result in Out
    store_result(Out_ptr, acc, l_i, m_i, stride_on)
