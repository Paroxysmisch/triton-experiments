device_ptr = torch.randn((1024, 1024), device='cuda')
stride = (512, 512)
blocksize = (32, 32)
groupsize = (16, 32)
