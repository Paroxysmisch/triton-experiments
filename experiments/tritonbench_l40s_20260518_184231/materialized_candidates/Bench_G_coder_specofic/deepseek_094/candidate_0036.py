import triton
import numpy as np

def get_score(Q, K, M, seq_len, dim, batch_size):
    Out = np.zeros((batch_size, seq_len), dtype=np.float32)
    sm_scale = 1.0 / sqrt(dim)

    grid_x = batch_size * seq_len
    block_x = 1024

    while True:
        try:
            _score_kernel[grid_x, block_x](Q, K, M, Out, batch_size, seq_len, dim, sm_scale)
            break
        except triton.language.cuda.CudaRuntimeError as e:
            if 'resource constraint' in str(e):
                if BLOCK_M > 1 and BLOCK_N > 1:
                    BLOCK_M //= 2
                    BLOCK_N //= 2
                else:
                    raise e

    return Out
