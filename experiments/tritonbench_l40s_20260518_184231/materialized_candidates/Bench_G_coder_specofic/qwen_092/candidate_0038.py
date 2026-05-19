triton
#include <triton/triton.h>

#define BLOCK_SIZE 128

__global__ void cross_entropy_fwd_kernel(
    const float* logits,
    const int* labels,
    float* loss,
    float* lse,
    float* z_loss,
    float smoothing,
    float logit_scale,
    float lse_square_scale,
    int ignored_index,
    int total_classes,
    int class_start_idx,
    int num_rows,
    int num_columns,
    bool HAS_SMOOTHING,
    bool SPLIT
) {
    __shared__ float shared_logits[BLOCK_SIZE];
    __shared__ float shared_lse[BLOCK_SIZE];
    __shared__ float shared_z_loss[BLOCK_SIZE];

    int row = tl.program_id(0);
    int block_col = tl.program_id(1);
    int col = block_col * BLOCK_SIZE + tl.program_id(2);

    float local_loss = 0.0f;
    float local_lse = -FLT_MAX;
    float local_z_loss = 0.0f;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        int idx = row * num_columns + col + i;
        if (col + i < num_columns) {
            shared_logits[i] = logits[idx] * logit_scale;
            if (col + i == labels[row]) {
                local_loss = -shared_logits[i];
            }
            local_lse = fmax(local_lse, shared_logits[i]);
        } else {
            shared_logits[i] = -FLT_MAX;
        }
    }

    tl.barrier();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tl.program_id(2) < stride) {
            shared_logits[tl.program_id(2)] = fmax(shared_logits[tl.program_id(2)], shared_logits[tl.program_id(2) + stride]);
            local_loss = fmax(local_loss, shared_loss[tl.program_id(2)]);
            local_lse = fmax(local_lse, shared_lse[tl.program_id(2)]);
        }
        tl.barrier();
    }

    if (tl.program_id(2) == 0) {
        shared_lse[0] = local_lse;
        shared_loss[0] = local_loss;
    }

    tl.barrier();

    if (col < num_columns) {
        if (labels[row] == ignored_index) {
            loss[row] = 0.0f;
            z_loss[row] = 0.0f;
        } else {
            loss[row] = shared_loss[0];
            z_loss[row] = shared_lse[0] * lse_square_scale;
        }
    }

    if (HAS_SMOOTHING && col < num_columns) {
        if (labels[row] != ignored_index) {
            loss[row] += smoothing / total_classes;
            z_loss[row] += smoothing / total_classes * shared_lse[0] * lse_square_scale;
        }
    }

    if (SPLIT && col < num_columns) {
        if (labels[row] != ignored_index) {
            loss[row] -= smoothing / total_classes;
            z_loss[row] -= smoothing / total_classes * shared_lse[0] * lse_square_scale;
        }
    }

    if (col < num_columns) {
        if (labels[row] != ignored_index) {
            loss[row] -= smoothing / total_classes;
            z_loss[row] -= smoothing / total_classes * shared_lse[0] * lse_square_scale;
        }
    }
}
