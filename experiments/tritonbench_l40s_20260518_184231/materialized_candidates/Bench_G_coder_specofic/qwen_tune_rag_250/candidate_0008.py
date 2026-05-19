triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim, 
        num_warps=num_warps,
        num_stages=1,
    )
    return
