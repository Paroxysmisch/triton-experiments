c++
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define WARP_SIZE 32

#define CHECK_CUDA(call) {                                                        \
  cudaError_t err = (call);                                                       \
  if (err != cudaSuccess) {                                                       \
    printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(-1);                                                                     \
  }                                                                                \
}

template <typename T>
__global__ void cholesky_kernel(const T* A, T* L, int n) {
  // 2D block and thread indices
  int block_y = blockIdx.y;
  int block_x = blockIdx.x;
  int thread_y = threadIdx.y;
  int thread_x = threadIdx.x;

  // Compute the global index for the matrix element
  int i = block_y * blockDim.y + thread_y;
  int j = block_x * blockDim.x + thread_x;

  // Check if we are within the matrix bounds
  if (i < n && j < n) {
    // Initialize the sum to 0
    T sum = static_cast<T>(0);

    // Compute the sum over the lower triangular part of the matrix
    for (int k = 0; k < j; ++k) {
      sum += L[i * n + k] * L[j * n + k];
    }

    // Compute the matrix element using the formula for the Cholesky decomposition
    if (i == j) {
      L[i * n + j] = sqrt(A[i * n + j] - sum);
    } else if (i > j) {
      L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
    } else {
      L[i * n + j] = static_cast<T>(0);
    }
  }
}

void cholesky(const float* A, float* L, int n) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
  cholesky_kernel<<<grid_dim, block_dim>>>(A, L, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void cholesky(const double* A, double* L, int n) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
  cholesky_kernel<<<grid_dim, block_dim>>>(A, L, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void cholesky(const cfloat* A, cfloat* L, int n) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
  cholesky_kernel<<<grid_dim, block_dim>>>(A, L, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void cholesky(const cdouble* A, cdouble* L, int n) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
  cholesky_kernel<<<grid_dim, block_dim>>>(A, L, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}
