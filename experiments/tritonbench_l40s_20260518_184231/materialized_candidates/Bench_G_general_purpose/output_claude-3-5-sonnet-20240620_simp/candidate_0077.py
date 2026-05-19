input_tensor = torch.randn(1000, 1000, device='cuda')
dropout_prob = 0.5
seed = 42

output = seeded_dropout(input_tensor, dropout_prob, seed)
