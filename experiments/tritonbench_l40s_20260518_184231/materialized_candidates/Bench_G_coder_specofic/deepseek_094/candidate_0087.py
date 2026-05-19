import triton

# Define your input tensors
y = triton.Tensor(shape=(1024, 1024), dtype=triton.float32)
gt = triton.Tensor(shape=(1024, 1024), dtype=triton.float32)

# Compute the KL divergence
loss = kldiv_forward_triton(y, gt, reduction='mean')

# Print the result
print(loss)
