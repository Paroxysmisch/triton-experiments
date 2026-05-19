b = torch.randn(2, 3, 4, 5, device="cuda")
print(mean_dim(b, [1, 2]))
