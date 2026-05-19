import triton
import triton.language as tl

# ---------------------------------------------
# Triton Kernel for Adam
# ---------------------------------------------
@triton.jit
def _adam_kernel(
    param_ptr,       # pointer to parameters
    grad_ptr,        # pointer to gradients
    m_ptr,           # pointer to first moment buffer
    v_ptr,           # pointer to second moment buffer
    vhat_ptr,        # pointer to Vhat buffer if AMSGrad=True, else None
    n_elements,      # total number of elements
    lr,              # learning rate
    beta1,           # beta1 for Adam
    beta2,           # beta2 for Adam
    eps,             # epsilon for numerical stability
    weight_decay,    # weight decay
    step,            # current iteration step (for bias correction)
    amsgrad,         # whether AMSGrad variant is used
    maximize,        # whether we should maximize instead of minimize
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements

    # Load data
    param = tl.load(param_ptr + block_start, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + block_start, mask=mask, other=0.0)
    m = tl.load(m_ptr + block_start, mask=mask, other=0.0)
    v = tl.load(v_ptr + block_start, mask=mask, other=0.0)
    if amsgrad:
        vhat = tl.load(vhat_ptr + block_start, mask=mask, other=0.0)
    else:
        vhat = 0.0  # not used if amsgrad=False

    # Optionally add weight decay
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    # If maximize=True, invert the sign of grad
    if maximize:
        grad = -grad

    # Update first moment m and second moment v
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)

    # Compute bias corrections
    one = tl.float32(1.0)
    bias_correction1 = one - beta1 ** step
    bias_correction2 = one - beta2 ** step

    # If AMSGrad: maintain the max of all second moment values
    if amsgrad:
        vhat = tl.max(vhat, v)
        denom = tl.sqrt(vhat / bias_correction2) + eps
    else:
        denom = tl.sqrt(v / bias_correction2) + eps

    # Compute step size
    step_size = lr * (tl.sqrt(bias_correction2) / bias_correction1)
    # Update param
    param = param - step_size * (m / denom)

    # Store results
    tl.store(param_ptr + block_start, param, mask=mask)
    tl.store(m_ptr + block_start, m, mask=mask)
    tl.store(v_ptr + block_start, v, mask=mask)
    if amsgrad:
        tl.store(vhat_ptr + block_start, vhat, mask=mask)

# ---------------------------------------------
# Adam Wrapper Function
# ---------------------------------------------
class Optimizer:
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, amsgrad=False, foreach=None,
        maximize=False, capturable=False, differentiable=False, fused=None
    ):
        """
        Functional Description:
            Implements the Adam optimization algorithm with optional AMSGrad,
            weight decay, and objective maximization. Supports various
            performance-oriented options (foreach, fused).
        """
        self.params = params  # List of parameter dicts: { 'param': ..., 'grad': ..., 'm': ..., 'v': ..., 'vhat': optional }
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.step_num = 0

    def step(self):
        """
        Perform a single optimization step with Adam.
        """
        self.step_num += 1

        # Decide block size (simple heuristic)
        BLOCK_SIZE = 1024

        for p_dict in self.params:
            param = p_dict['param']
            grad = p_dict['grad']
            m    = p_dict['m']
            v    = p_dict['v']
            vhat = p_dict.get('vhat', None)

            n_elements = param.numel()

            # Launch kernel
            grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
            triton.run(
                _adam_kernel,
                grid=grid,
                num_warps=4,
                BLOCK_SIZE=BLOCK_SIZE,
                param_ptr=param,
                grad_ptr=grad,
                m_ptr=m,
                v_ptr=v,
                vhat_ptr=vhat if vhat is not None else grad,  # dummy if None
                n_elements=n_elements,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                step=self.step_num,
                amsgrad=self.amsgrad,
                maximize=self.maximize
            )

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
         foreach=None, maximize=False, capturable=False, differentiable=False, fused=None) -> Optimizer:
    """
    Adam Optimizer
    Functional Description:
        Implements Adam algorithm with optional AMSGrad, weight decay,
        and objective maximization. The foreach and fused implementations
        are typically faster than the for-loop, single-tensor implementation.
    Math:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """
    return Optimizer(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=foreach,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused
    )
