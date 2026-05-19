statistic < 0.01

# Function to test normal PRNG
def test_randn(size, seed, device='cuda'):
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK),)
    kernel_randn[grid](x, N, seed)
    assert abs(x.mean()) < 1e-2
    assert abs(x.std() - 1) < 1e-2

# Function to test rand limits
def test_rand_limits():
    min_max_int32 = torch.tensor([
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
    ], dtype=torch.int32, device='cuda')
    output = torch.empty(2, dtype=torch.float32, device='cuda')
    kernel_rand_limits[(1,)](min_max_int32, output, 2)
    assert output[0] == output[1]
    assert 1.0 - torch.finfo(torch.float32).eps <= output[0].item() < 1.0
