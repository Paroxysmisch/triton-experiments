@triton.jit
def logspace(T_start, T_end, steps, base=10.0):
    # Define the function signature
    func_inputs = (T_start, T_end, steps, base)
    # Generate the tensor values using logarithmic progression
    tensor_values = torch.logspace(start=T_start, end=T_end, steps=steps, base=base)
    return tensor_values
