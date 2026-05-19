c++
__device__ void _layer_norm_forward_kernel(float* X, float* W, float* B, float* Y, float* Mean, float* RSTD, int num_features, int batch_size) {
    // Load input, weight, and bias
    float x = X[blockIdx.x * num_features + threadIdx.x];
    float w = W[threadIdx.x];
    float b = B[threadIdx.x];

    // Compute mean and variance
    float mean = tl::sum(X, batch_size) / batch_size;
    float variance = tl::sum((X - mean) * (X - mean), batch_size) / batch_size;

    // Normalize input and store in Y
    Y[blockIdx.x * num_features + threadIdx.x] = (x - mean) / sqrt(variance + 1e-5);
    Y[blockIdx.x * num_features + threadIdx.x] = Y[blockIdx.x * num_features + threadIdx.x] * w + b;

    // Store mean and RSTD
    Mean[blockIdx.x] = mean;
    RSTD[blockIdx.x] = sqrt(variance + 1e-5);
}

__device__ void _layer_norm_backward_kernel(float* X, float* W, float* B, float* DX, float* DW, float* DB, float* Mean, float* RSTD, int num_features, int batch_size) {
    // Load input, weight, and bias
    float x = X[blockIdx.x * num_features + threadIdx.x];
    float w = W[threadIdx.x];
    float b = B[threadIdx.x];
    float mean = Mean[blockIdx.x];
    float rstd = RSTD[blockIdx.x];

    // Compute gradients
    float dx = (1.0 / batch_size) * rstd;
    float dw = (x - mean) * dx;
    float db = dx;

    // Store gradients
    DX[blockIdx.x * num_features + threadIdx.x] = dx;
    DW[threadIdx.x] = dw;
    DB[threadIdx.x] = db;
}
