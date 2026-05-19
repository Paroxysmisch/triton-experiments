import triton

            def add_kernel_fn(a, b, c):
                with triton.cuda_context():
                    kernel = triton.TritonKernel(add_kernel)
                    kernel.launch(grid=(len(c),), block=(len(c),), shared=0, args=(a, b, c))
