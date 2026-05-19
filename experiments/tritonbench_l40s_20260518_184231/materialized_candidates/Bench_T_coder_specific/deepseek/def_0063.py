a = triton.random.randn(2, 3, 4)
b = triton.random.randn(4, 5)

# Contract the last two dimensions of a with the first dimension of b
result = triton.tensordot(a, b, dims=(3, 0))

# Contract the last dimension of a with the first dimension of b
result = triton.tensordot(a, b, dims=1)

# Contract the last two dimensions of a with the last two dimensions of b
result = triton.tensordot(a, b, dims=((3,), (0, 1)))
