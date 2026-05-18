# TritonBench L40S Run

VM: gxp-l40s-2
Date: 2026-05-18T18:42:34+01:00
Branch: mantas-tritonbench

## GPU

```
name, driver_version, memory.total [MiB]
NVIDIA L40S, 590.48.01, 46068 MiB
NVIDIA L40S, 590.48.01, 46068 MiB
NVIDIA L40S, 590.48.01, 46068 MiB
NVIDIA L40S, 590.48.01, 46068 MiB
```

## Python

```
torch 2.5.1+cu124
triton 3.1.0
cuda available True
device count 1
device 0 NVIDIA L40S
```

## Performance Summary

- TritonBench-G: speed up 0.81, efficiency 28.25
- TritonBench-T: speed up 0.66
- T rejected slow outliers: eig.json at 0.046x, spectral_norm_eig.json at 0.0519x

## I/O Summary

```json
{
  "G": {
    "total": 184,
    "failed": 13,
    "failures": [
      {
        "file": "attention_forward_triton.py",
        "log": "benchmark_runs/io_check/G_attention_forward_triton.py.log"
      },
      {
        "file": "attention_fwd_triton1.py",
        "log": "benchmark_runs/io_check/G_attention_fwd_triton1.py.log"
      },
      {
        "file": "attention_kernel.py",
        "log": "benchmark_runs/io_check/G_attention_kernel.py.log"
      },
      {
        "file": "attn_fwd_causal.py",
        "log": "benchmark_runs/io_check/G_attn_fwd_causal.py.log"
      },
      {
        "file": "attn_fwd_triton.py",
        "log": "benchmark_runs/io_check/G_attn_fwd_triton.py.log"
      },
      {
        "file": "chunk_retention.py",
        "log": "benchmark_runs/io_check/G_chunk_retention.py.log"
      },
      {
        "file": "chunk_retention_ops.py",
        "log": "benchmark_runs/io_check/G_chunk_retention_ops.py.log"
      },
      {
        "file": "lightning_attention.py",
        "log": "benchmark_runs/io_check/G_lightning_attention.py.log"
      },
      {
        "file": "matmul_dequantize.py",
        "log": "benchmark_runs/io_check/G_matmul_dequantize.py.log"
      },
      {
        "file": "matmul_kernel.py",
        "log": "benchmark_runs/io_check/G_matmul_kernel.py.log"
      },
      {
        "file": "matmul_persistent_triton.py",
        "log": "benchmark_runs/io_check/G_matmul_persistent_triton.py.log"
      },
      {
        "file": "streamk_matmul.py",
        "log": "benchmark_runs/io_check/G_streamk_matmul.py.log"
      },
      {
        "file": "token_attn_reduceV.py",
        "log": "benchmark_runs/io_check/G_token_attn_reduceV.py.log"
      }
    ]
  },
  "T": {
    "total": 166,
    "failed": 1,
    "failures": [
      {
        "file": "quantize_dynamic.py",
        "log": "benchmark_runs/io_check/T_quantize_dynamic.py.log"
      }
    ]
  }
}
```
