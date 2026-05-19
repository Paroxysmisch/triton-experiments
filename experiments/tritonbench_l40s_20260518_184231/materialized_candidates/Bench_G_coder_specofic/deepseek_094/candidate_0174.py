c++
__device__ float dot_product(float* a, float* b, int len) {
    float result = 0.0f;
    for (int i = 0; i < len; i++) {
        result += a[i] * b[i];
    }
    return result;
}

__global__ void _fwd_kernel_token_att1(
    float* Q, float* K, float* Att_Out, 
    int* B_Loc, int* B_Start_Loc, int* B_Seqlen, 
    int max_input_len, float sm_scale, 
    int num_heads, int block_size) 
{
    int batch_id = blockIdx.x / num_heads;
    int head_id = blockIdx.x % num_heads;
    int block_id = threadIdx.x / block_size;
    int block_offset = threadIdx.x % block_size;

    int seq_start = B_Start_Loc[batch_id];
    int seq_len = B_Seqlen[batch_id];
    int key_dim = B_Loc[block_id * max_input_len + seq_start + block_offset];

    float* q_segment = &Q[head_id * seq_len * key_dim + block_id * block_size * key_dim];
    float* k_segment = &K[head_id * seq_len * key_dim + block_offset];
    float* att_out_segment = &Att_Out[head_id * seq_len * seq_len + block_id * seq_len + block_offset];

    float dot_prod = dot_product(q_segment, k_segment, key_dim);
    *att_out_segment = dot_prod * sm_scale;
}

void token_att_fwd(
    float* Q, float* K, float* Att_Out, 
    int* B_Loc, int* B_Start_Loc, int* B_Seqlen, 
    int max_input_len, float sm_scale, 
    int num_heads, int block_size, 
    cudaStream_t stream) 
{
    dim3 grid_dim(num_heads * max_input_len / block_size);
    dim3 block_dim(block_size);

    _fwd_kernel_token_att1<<<grid_dim, block_dim, 0, stream>>>(
        Q, K, Att_Out, 
        B_Loc, B_Start_Loc, B_Seqlen, 
        max_input_len, sm_scale, 
        num_heads, block_size);
}
