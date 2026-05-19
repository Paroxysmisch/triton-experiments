cpp
#include <torch/torch.h>

torch::Tensor tril_mm_and_scale_kernel(torch::Tensor A, torch::Tensor B, float alpha, float beta) {
    auto tri_A = torch::tril(A);
    auto mul_result = torch::mm(tri_A, B);
    auto scaled_mul_result = mul_result * alpha;
    auto final_result = scaled_mul_result * beta;
    return final_result;
}
