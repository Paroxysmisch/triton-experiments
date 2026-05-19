def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the cosine of each element in the input tensor
    cos_result = torch.cos(input)
    
    # Determine the sign bit for each cosine result
    sign_bit = torch.signbit(cos_result)
    
    return cos_result, sign_bit
