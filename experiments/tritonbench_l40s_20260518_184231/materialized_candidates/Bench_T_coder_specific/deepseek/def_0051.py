def cos_avg_pool1d_twice(input: torch.Tensor, kernel_size1: int, stride1: int = None, padding1: int = 0, ceil_mode1: bool = False, count_include_pad1: bool = True, kernel_size2: int, stride2: int = None, padding2: int = 0, ceil_mode2: bool = False, count_include_pad2: bool = True) -> torch.Tensor:
    # Compute the cosine of each element in the input tensor
    input = torch.cos(input)

    # Apply 1D average pooling with the first set of parameters
    input = F.avg_pool1d(input, kernel_size1, stride1, padding1, ceil_mode1, count_include_pad1)

    # Apply 1D average pooling with the second set of parameters
    output = F.avg_pool1d(input, kernel_size2, stride2, padding2, ceil_mode2, count_include_pad2)

    return output
