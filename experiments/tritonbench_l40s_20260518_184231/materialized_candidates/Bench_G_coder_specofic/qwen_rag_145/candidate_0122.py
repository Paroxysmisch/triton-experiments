class Fast_RoPE_Embedding(torch.autograd.Function): 
	... 
	Where you put the Python wrapper functions for applying the Triton kernel for forward and backward pass with the necessary steps.

def fast_rope_embedding(Q, K, cos, sin): 
	... 
	Where you put the final function to apply the Fast_RoPE_Embedding to both Q and K.
