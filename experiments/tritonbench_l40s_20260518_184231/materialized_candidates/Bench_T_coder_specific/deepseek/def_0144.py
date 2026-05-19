cpp
extern "C" {
    void matrix_multiply_and_row_dot(const float* A, const float* B, float alpha, float beta, float* C, int n, int m, int p) {
        // Perform matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, alpha, A, m, B, p, beta, C, p);

        // Compute dot product of the first two rows
        float dot_product = cblas_ddot(p, C, 1, C+p, 1);

        // You might want to return the dot product somehow...
    }
}
