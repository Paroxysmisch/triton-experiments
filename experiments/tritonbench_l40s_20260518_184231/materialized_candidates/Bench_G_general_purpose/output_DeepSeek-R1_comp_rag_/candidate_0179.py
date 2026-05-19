col_idx = tl.program_id(0) * BLOCK_SIZE  # Block index -> data offset
     col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)  # Thread indices within block
     mask = col_offsets < batch_size * dim  # Boundary check
     
       new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
       new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag
       
       grad_s = grad_y + grad_s * Lambda
       grad_Lambda += grad_s * s_prev
       
       grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
       grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
       
   grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
   diag_ssm_forward_kernel[grid](...)  # Real or complex version
   
   diag_ssm_backward_kernel[grid](...)
   grad_lambda.sum(dim=0)  # Sum over batch dimension
