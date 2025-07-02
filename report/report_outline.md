# CUDA vs Triton: Kernel Performance Report

## 1. Model Selection
- Chosen: DistilBERT
- Justification: Lightweight, Transformer-based, matmul-heavy

## 2. Baseline Inference with cuDNN (PyTorch)
- Summary of how inference was run
- Nsight metrics: latency, memory, GPU%, kernel list

## 3. Triton Kernel Test
- Operation: MatMul
- Kernel reused from Triton tutorial
- Performance metrics from Nsight

## 4. Comparison Table

| Metric        | cuDNN        | Triton       |
|---------------|--------------|--------------|
| Latency       |              |              |
| Memory Usage  |              |              |
| GPU Util %    |              |              |

## 5. Assumptions & Limitations
- No access to Blackwell GPU
- Used static inputs, no fine-tuning
- Triton used on isolated kernel

## 6. Conclusion
- Insights
- Which library performed better for what