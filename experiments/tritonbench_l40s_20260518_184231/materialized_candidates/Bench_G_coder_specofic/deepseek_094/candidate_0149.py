cpp
__device__
void update_fn_kernel(float* p_ptr, float* grad_ptr, float* exp_avg_ptr, int pid, int n_elements, float lr, float wd, float beta1, float beta2) {
    int tid = threadIdx.x;
    int block_start = pid * BLOCK_SIZE;
    if (block_start + tid >= n_elements) return;

    float param = p_ptr[block_start + tid];
    float grad = grad_ptr[block_start + tid];
    float exp_avg = exp_avg_ptr[block_start + tid];

    float new_param = param * (1 - lr * wd);
    float diff = exp_avg - grad;
    float update = beta1 * diff;
    bool sign_change = (sign(param) != sign(update));

    if (sign_change) {
        new_param = param - lr * update;
    } else {
        new_param = param - lr * update;
    }

    exp_avg = beta2 * exp_avg + (1 - beta2) * grad;

    p_ptr[block_start + tid] = new_param;
    exp_avg_ptr[block_start + tid] = exp_avg;
}

void update_fn(torch::Tensor p, torch::Tensor grad, torch::Tensor exp_avg, float lr, float wd, float beta1, float beta2) {
    const int threads = BLOCK_SIZE;
    const int blocks = p.numel() / threads;

    AT_DISPATCH_FLOATING_TYPES(p.type(), "update_fn", ([&] {
        update_fn_kernel<<<blocks, threads>>>(
            p.data_ptr<float>(), 
            grad.data_ptr<float>(), 
            exp_avg.data_ptr<float>(), 
            blocks, 
            p.numel(), 
            lr, 
            wd, 
            beta1, 
            beta2
        );
    }));
}
