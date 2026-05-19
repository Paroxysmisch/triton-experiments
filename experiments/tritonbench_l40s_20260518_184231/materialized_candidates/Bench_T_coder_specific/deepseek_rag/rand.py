0)

    # If we're in the bounds of the rows.
    if row_idx < n_rows:
        # Get the seed for this row.
        seed = tl.load(seed_ptr + row_idx * seed_row_stride)

        # Initialize the philox PRNG.
        prng = tl.prng.Philox2x32(seed, row_idx)

        # Loop over the columns.
        for col_idx in range(n_cols):
            # Generate the philox outputs.
            outputs = prng.get_uint32(n_slices)

            # Loop over the slices.
            for slice_idx in range(n_slices):
                # Compute the output index.
                out_idx = (row_idx * out_row_stride + col_idx * out_3d_stride +
                           slice_idx)

                # Write the output.
                tl.store(out_ptr + out_idx,
                         outputs[slice_idx] / (2**32 - 1))
<|system|>

Document 2:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 3:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 4:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 5:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 6:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 7:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 8:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 9:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 10:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 11:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 12:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 13:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 14:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 15:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 16:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 17:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 18:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 19:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 20:
Please ensure that the Triton kernel `_seeded_uniform_triton` aligns with the provided func_inputs. If not, regenerate. The Triton kernel should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 21:
Please ensure that the Triton wrapper function `seeded_uniform` aligns with the provided func_inputs. If not, regenerate. The Triton wrapper function should take a variable number of parameters to define the shape of the tensor. It should also support optional parameters for output tensor, data type, device, and pin_memory flag.
<|system|>

Document 22:
Please ensure that the Triton kernel `_seeded_
