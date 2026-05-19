Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda')
cos = torch.cos(position_ids)
sin = torch.sin(position_ids)
output = rope_embedding(Q, cos, sin)
