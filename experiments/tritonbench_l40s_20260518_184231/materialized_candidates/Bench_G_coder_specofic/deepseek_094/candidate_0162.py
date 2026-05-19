cpp
__device__
void matmul_kernel(int m_size, int k_size, int n_size, int m_block_size, int k_block_size, int n_block_size,
                   const float* x, const float* y, float* z, int pid) {
    int m_block_id = pid / (k_size / m_block_size * n_size / n_block_size);
    int k_block_id = (pid % (k_size / m_block_size * n_size / n_block_size)) / (n_size / n_block_size);
    int n_block_id = (pid % (k_size / m_block_size * n_size / n_block_size)) % (n_size / n_block_size);

    __shared__ float x_block[m_block_size][k_block_size];
    __shared__ float y_block[k_block_size][n_block_size];

    for (int m = 0; m < m_size; m += m_block_size) {
        for (int k = 0; k < k_size; k += k_block_size) {
            for (int n = 0; n < n_size; n += n_block_size) {
                int m_start = m + m_block_id * m_block_size;
                int k_start = k + k_block_id * k_block_size;
                int n_start = n + n_block_id * n_block_size;

                if (threadIdx.x < m_block_size && threadIdx.y < k_block_size) {
                    x_block[threadIdx.x][threadIdx.y] = x[m_start * k_size + k_start * k_size + threadIdx.x * k_size + threadIdx.y];
                }
                if (threadIdx.x < k_block_size && threadIdx.y < n_block_size) {
                    y_block[threadIdx.x][threadIdx.y] = y[k_start * n_size + n_start * k_size + threadIdx.x * n_size + threadIdx.y];
                }
                __syncthreads();

                float z_val = 0.0f;
                for (int i = 0; i < k_block_size; ++i) {
                    z_val += x_block[threadIdx.x][i] * y_block[i][threadIdx.y];
                }
                if (threadIdx.x < m_block_size && threadIdx.y < n_block_size) {
                    z[m_start * n_size + n_start * m_size + threadIdx.x * n_size + threadIdx.y] = z_val;
                }
                __syncthreads();
            }
        }
    }
}

void matmul(int m_size, int k_size, int n_size, int m_block_size, int k_block_size, int n_block_size,
            const float* x, const float* y, float* z) {
    dim3 grid((k_size / m_block_size * n_size / n_block_size + blockSize - 1) / blockSize);
    dim3 block(m_block_size, n_block_size);

    for (int i = 0; i < m_size; ++i) {
        for (int j = 0; j < n_size; ++j) {
            z[i * n_size + j] = 0.0f;
        }
    }

    for (int pid = 0; pid < grid.x * block.x * block.y; ++pid) {
        matmul_kernel<<<grid, block>>>(m_size, k_size, n_size, m_block_size, k_block_size, n_block_size, x, y, z, pid);
    }
}
