out_rms_triton = rmsnorm_triton_wrapper(x=embeddings_load, rms_w=rms_weights)
  
  out_rbe_triton = rbe_triton_wrapper(xq_output_triton, pos=0).view(batch, seq_len, heads, dim)
  
  out_rms_matmul_rbe_triton = rms_matmul_rbe_wrapper(
      x=embeddings_load, start_pos=0, weight=q_weights_load, rms_w=rms_weights,
      use_rbe=True, n_heads=32, head_dim=128).view(batch, seq_len, heads, dim)
  
  out_rms_matmul_rbe_qkv, _, _ = rms_matmul_rbe_qkv_wrapper(
      x=embeddings_load, start_pos=0, q_weight=q_weights_load, k_weight=q_weights_load,
      v_weight=q_weights_load, rms_w=rms_weights, k=k, v=v,
      n_heads=32, head_dim=128)
