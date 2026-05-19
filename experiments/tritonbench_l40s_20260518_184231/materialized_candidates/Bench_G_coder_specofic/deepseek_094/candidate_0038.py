c++
// Include Triton header
#include <triton/kernel_decl.h>

// Forward pass kernel
__device__ void cross_entropy_fwd_kernel(const float* __restrict__ logits,
                                         const int* __restrict__ labels,
                                         float* __restrict__ loss,
                                         float* __restrict__ lse,
                                         float* __restrict__ z_loss,
                                         float smoothing,
                                         float logit_scale,
                                         float lse_square_scale,
                                         int ignored_index,
                                         int total_classes,
                                         int class_start_idx,
                                         bool HAS_SMOOTHING,
                                         bool SPLIT,
                                         int row,
                                         int col) {
    // Kernel implementation here
}

// Backward pass kernel
__device__ void cross_entropy_bwd_kernel(const float* __restrict__ logits,
                                         const int* __restrict__ labels,
                                         float* __restrict__ dlogits,
                                         float lse,
                                         float z_loss,
                                         float smoothing,
                                         float logit_scale,
                                         int ignored_index,
                                         int total_classes,
                                         int class_start_idx,
                                         bool HAS_SMOOTHING,
                                         int row,
                                         int col) {
    // Kernel implementation here
}

// Forward pass wrapper function
void cross_entropy_fwd(const Tensor& logits,
                       const Tensor& labels,
                       float smoothing,
                       float logit_scale,
                       float lse_square_scale,
                       int ignored_index,
                       int total_classes,
                       int class_start_idx,
                       bool HAS_SMOOTHING,
                       bool SPLIT,
                       Tensor& loss,
                       Tensor& lse,
                       Tensor& z_loss) {
    // Kernel configuration and dispatch here
}

// Backward pass wrapper function
void cross_entropy_bwd(const Tensor& logits,
                       const Tensor& labels,
                       float lse,
                       float z_loss,
                       float smoothing,
                       float logit_scale,
                       int ignored_index,
                       int total_classes,
                       int class_start_idx,
                       bool HAS_SMOOTHING,
                       Tensor& dlogits) {
    // Kernel configuration and dispatch here
}
