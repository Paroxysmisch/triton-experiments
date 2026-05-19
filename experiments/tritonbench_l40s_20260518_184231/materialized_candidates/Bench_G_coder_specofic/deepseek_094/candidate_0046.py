cpp
#include <triton/api.h>

void _l2_norm_fwd_1pass_kernel(
    float* Y, const float* X, 
    const uint32_t num_rows, 
    const uint32_t num_cols) {
    uint32_t row = tl::program_id(0);
    if (row >= num_rows) return;

    float sum_of_squares = 0.0f;
    for (uint32_t col = 0; col < num_cols; ++col) {
        float val = X[row * num_cols + col];
        sum_of_squares += val * val;
    }

    float rstd = 1.0f / sqrtf(sum_of_squares);
    for (uint32_t col = 0; col < num_cols; ++col) {
        float val = X[row * num_cols + col];
        Y[row * num_cols + col] = val * rstd;
    }
}

void _l2_norm_bwd_kernel(
    float* DX, const float* X, const float* DY, 
    const float* rstd, 
    const uint32_t num_rows, 
    const uint32_t num_cols) {
    uint32_t row = tl::program_id(0);
    if (row >= num_rows) return;

    for (uint32_t col = 0; col < num_cols; ++col) {
        float val = X[row * num_cols + col];
        float dy = DY[row * num_cols + col];
        DX[row * num_cols + col] = dy * rstd[row] * (val - rstd[row] * sum_of_squares);
    }
}

void _l2_norm_fwd(
    Tensor* Y, const Tensor* X, 
    Tensor* rstd) {
    uint32_t num_rows = X->shape[0];
    uint32_t num_cols = X->shape[1];

    // Prepare Y
    Y->reshape({num_rows, num_cols});
    // Prepare rstd
    rstd->reshape({num_rows});

    // Run kernel
    triton::api::launch(_l2_norm_fwd_1pass_kernel,
        Y->data, X->data, 
        num_rows, num_cols,
        rstd->data);
}

void _l2_norm_bwd(
    Tensor* DX, const Tensor* X, const Tensor* DY, 
    const Tensor* rstd) {
    uint32_t num_rows = X->shape[0];
    uint32_t num_cols = X->shape[1];

    // Prepare DX
    DX->reshape({num_rows, num_cols});

    // Run kernel
    triton::api::launch(_l2_norm_bwd_kernel,
        DX->data, X->data, DY->data, 
        rstd->data, 
        num_rows, num_cols);
}
