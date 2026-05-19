cpp
// Calculate the optimal block size and number of warps
void calculate_settings(int n, int& num_warps, int& block_size) {
    // ...
}

// Forward pass
__global__ void layernorm_forward(float* X, float* W, float* b, float* Y, float* r, float* mu, int n) {
    // ...
}

// Backward pass
__global__ void layernorm_backward(float* X, float* dY, float* W, float* b, float* dX, float* r, float* mu, int n) {
    // ...
}
