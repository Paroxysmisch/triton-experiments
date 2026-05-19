import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 1024

@triton.jit
def calculate_settings(n: tl.int32) -> tl.int32:
    # Determine the optimal block size and number of warps
    block_size = min(n, MAX_FUSED_SIZE)
    num_warps = (block_size + 31) // 32
    return block_size, num_warps

@triton.jit
def _cross_entropy_forward(
    logits: tl.tensor, labels: tl.tensor, dlosses: tl.tensor,
    n: tl.int32, valid_labels: tl.tensor, softcapping: tl.float32, logit_scale: tl.float32,
    log_sum_exp: tl.tensor, loss: tl.tensor, block_size: tl.int32, num_warps: tl.int32
):
    pid = tl.program_id(0)
    coords = pid * block_size + tl.arange(0, block_size)
    valid = coords < n
    coords = tl.where(valid, coords, -1)
    labels = tl.where(valid, labels, -100)
    logits = tl.where(valid, logits, -1e9)
    
    if softcapping > 0:
        logits = tl.clip(logits, -softcapping, softcapping)
    
    if logit_scale > 0:
        logits = logits * logit_scale
    
    exp_logits = tl.exp(logits)
    log_sum_exp[pid] = tl.sum(exp_logits)
    
    if valid_labels[pid] != -100:
        true_logit = exp_logits[valid_labels[pid]]
        loss[pid] = -tl.log(true_logit) + tl.log(log_sum_exp[pid])

@triton.jit
def _chunked_cross_entropy_forward(
    logits: tl.tensor, labels: tl.tensor, dlosses: tl.tensor,
    n: tl.int32, valid_labels: tl.tensor, softcapping: tl.float32, logit_scale: tl.float32,
    log_sum_exp: tl.tensor, loss: tl.tensor, chunk_size: tl.int32, num_warps: tl.int32
):
    pid = tl.program_id(0)
    coords = pid * chunk_size + tl.arange(0, chunk_size)
    valid = coords < n
    coords = tl.where(valid, coords, -1)
    labels = tl.where(valid, labels, -100)
    logits = tl.where(valid, logits, -1e9)
    
    if softcapping > 0:
        logits = tl.clip(logits, -softcapping, softcapping)
    
    if logit_scale > 0:
        logits = logits * logit_scale
    
    exp_logits = tl.exp(logits)
    log_sum_exp[pid] = tl.sum(exp_logits)
    
    if valid_labels[pid] != -100:
        true_logit = exp_logits[valid_labels[pid]]
        loss[pid] = -tl.log(true_logit) + tl.log(log_sum_exp[pid])

@triton.jit
def _cross_entropy_backward(
    logits: tl.tensor, dlosses: tl.tensor, grad: tl.tensor,
    n: tl.int32, valid_labels: tl.tensor, softcapping: tl.float32, logit_scale: tl.float32,
    block_size: tl.int32, num_warps: tl.int32
):
    pid = tl.program_id(0)
    coords = pid * block_size + tl.arange(0, block_size)
    valid = coords < n
    coords = tl.where(valid, coords, -1)
    labels = tl.where(valid, labels, -100)
    logits = tl.where(valid, logits, -1e9)
    
    if softcapping > 0:
        logits = tl.clip(logits, -softcapping, softcapping)
    
    if logit_scale > 0:
        logits = logits * logit_scale
    
    exp_logits = tl.exp(logits)
    softmax = exp_logits / tl.sum(exp_logits)
    
    grad[pid] = (softmax - tl.eye(n)[valid_labels[pid]]) * dlosses[pid]

class Fast_CrossEntropyLoss:
    def __init__(self, n, chunk_size=None):
        self.n = n
        self.chunk_size = chunk_size if chunk_size is not None else n
        self.block_size, self.num_warps = calculate_settings(self.n)
    
    def forward(self, logits, labels, valid_labels, softcapping=0.0, logit_scale=1.0):
        n = logits.shape[1]
        log_sum_exp = tl.zeros([logits.shape[0]], dtype=logits.dtype)
        loss = tl.zeros([logits.shape[0]], dtype=logits.dtype)
        
        if n <= MAX_FUSED_SIZE:
            _cross_entropy_forward(
                logits, labels, tl.zeros_like(logits), n, valid_labels, softcapping, logit_scale,
                log_sum_exp, loss, self.block_size, self.num_warps
            )
        else:
            _chunked_cross_entropy_forward(
                logits, labels, tl.zeros_like(logits), n, valid_labels, softcapping, logit_scale,
                log_sum_exp, loss, self.chunk_size, self.num_warps
            )
        
        return loss
    
    def backward(self, logits, dlosses, valid_labels, softcapping=0.0, logit_scale=1.0):
        n = logits.shape[1]
        grad = tl.zeros_like(logits)
        
        _cross_entropy_backward(
            logits, dlosses, grad, n, valid_labels, softcapping, logit_scale,
            self.block_size, self.num_warps
        )
        
        return grad

def fast_cross_entropy_loss(logits, labels, valid_labels, chunk_size=None, softcapping=0.0, logit_scale=1.0):
    n = logits.shape[1]
    cross_entropy_loss = Fast_CrossEntropyLoss(n, chunk_size)
    loss = cross_entropy_loss.forward(logits, labels, valid_labels, softcapping, logit_scale)
    mean_loss = tl.mean(loss)
    
    return mean_loss
