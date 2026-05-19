class FP8_Type:
    def __init__(self):
        self.num_bits = 16
        self.exponent_bits = 3
        self.mantissa_bits = 13
        self.sign_bit = 1
        self.max_val = (1 << (self.num_bits - self.exponent_bits - self.mantissa_bits - self.sign_bit)) - 2**-self.mantissa_bits - 1 
        self.min_val = -((1 << (self.num_bits - self.exponent_bits - self.mantissa_bits - 1)) - 1)
        self.smallest_normal = (1 << (self.exponent_bits - 1)) * (1 << (self.mantissa_bits - 1))

FP8 = FP8_Type()
