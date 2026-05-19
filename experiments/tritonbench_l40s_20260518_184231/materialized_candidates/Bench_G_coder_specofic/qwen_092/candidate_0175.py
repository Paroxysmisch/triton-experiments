cpp
#include <triton/core.hh>
#include <triton/ops.hh>

namespace triton {
namespace kernels {
namespace l2_norm_bwd {

// Kernel function to perform the backward pass of L2 normalization
__global__ void _l2_norm_bwd_kernel(
    const float* X,
    const float* DY,
    float* DX,
    const float* var,
    float eps,
    int M,
    int N,
    int stride_x_row) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  float sum = 0.0f;
  float x_dot_dy = 0.0f;

  // Compute the sum of DY * X and DY * DY for the current row
  for (int n = 0; n < N; ++n) {
    float x = X[row * stride_x_row + n];
    float dy = DY[row * stride_x_row + n];
    sum += dy * x;
    x_dot_dy += dy * dy;
  }

  // Compute the reciprocal of the standard deviation
  float rstd = 1.0f / std::sqrt(var[row] + eps);

  // Compute the gradient DX for the current row
  for (int n = 0; n < N; ++n) {
    float x = X[row * stride_x_row + n];
    DX[row * stride_x_row + n] = dy[row * stride_x_row + n] * rstd - sum * (1.0f / (var[row] + eps)) * rstd * x;
  }
}

// Triton operator function to orchestrate the backward pass of L2 normalization
void _l2_norm_bwd(
    const Tensor& x,
    const Tensor& dy,
    Tensor& dx,
    const Tensor& var,
    float eps) {
  int M = x.shape(0);
  int N = x.shape(1);

  // Ensure N is within the block size limit
  if (N > BLOCK_N) {
    throw std::runtime_error("Feature dimension N exceeds the maximum allowable fused size");
  }

  // Reshape and configure strides if necessary
  Tensor reshaped_x = x.contiguous();
  Tensor reshaped_dy = dy.contiguous();
  Tensor reshaped_dx = dx.contiguous();

  // Launch the kernel over M rows
  int blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  _l2_norm_bwd_kernel<<<blocks, BLOCK_SIZE>>>(reshaped_x.data<float>(), reshaped_dy.data<float>(), reshaped_dx.data<float>(), var.data<float>(), eps, M, N, reshaped_x.stride(1));

  // Reshape the output back to the original input shape
  dx = reshaped_dx.reshape({M, N});
}

}  // namespace l2_norm_bwd
}  // namespace kernels
}  // namespace triton
