def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    # Initialize parameters
    for param in params:
        state = {}
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param.data)
        state['exp_avg_sq'] = torch.zeros_like(param.data)
        if amsgrad:
            state['max_exp_avg_sq'] = torch.zeros_like(param.data)
        param.state = state

    def update(param, grad, state):
        state['step'] += 1
        bias_correction1 = 1 - betas[0] ** state['step']
        bias_correction2 = 1 - betas[1] ** state['step']

        # Update first and second moment running average
        state['exp_avg'] *= betas[0]
        state['exp_avg'] += (1 - betas[0]) * grad
        state['exp_avg_sq'] *= betas[1]
        state['exp_avg_sq'] += (1 - betas[1]) * grad ** 2

        if amsgrad:
            # Maintain max of 'max_exp_avg_sq' and 'exp_avg_sq'
            torch.max(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
            denom = (state['max_exp_avg_sq'] ** 0.5 + eps)
        else:
            denom = (state['exp_avg_sq'] ** 0.5 + eps)

        # Update parameters
        param.data -= lr * state['exp_avg'] / bias_correction1 / denom

    return update
