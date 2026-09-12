# FedGAPrompt: Minimum Requirements (Software & Hardware)

## Hardware Requirements

### GPU / VRAM

**Minimum** (4-bit quantized 7B/8B models, e.g. Qwen3-8B, Llama-3-8B):
- GPU: NVIDIA GPU with CUDA Compute Capability >= 7.0 (Turing or newer)
- VRAM: 8 GB VRAM (e.g. RTX 3060/4060, T4, RTX 2070/2080)
- Note: Ampere or newer (RTX 30xx+, A100) recommended for bfloat16.
  For older GPUs (Turing/Volta), set `compute_dtype="float16"`.

**Recommended** (14B models like Phi-4, or larger token outputs):
- VRAM: 16 GB - 24 GB VRAM (e.g. RTX 3090, RTX 4090, A4000, A5000, A100)

**Non-Quantized (FP16/BF16 full precision)**:
- 7B/8B: >= 16 GB VRAM
- 14B: >= 32 GB VRAM

### System Memory (RAM)
- Minimum: 16 GB RAM (for dataset caching and model weight loading)
- Recommended: 32 GB RAM (for seamless batch processing & multiple datasets)

### Storage / Disk
- Minimum: 30 GB free disk space (SSD recommended) for 1 base model checkpoint,
  embeddings model (all-MiniLM-L6-v2), datasets (MedQA/PubMedQA/MedMCQA).
- Recommended: 60+ GB free SSD space when evaluating multiple model architectures.

### CPU & OS
- CPU: 4 cores / 8 threads minimum (8+ cores recommended)
- OS: Linux (Ubuntu 20.04+) recommended with NVIDIA drivers >= 525 (CUDA 12.x+)

## Software Dependencies

To install:
```bash
pip install -r minimum_requirement.txt
```

### Core Deep Learning & LLM Inference
- torch>=2.0.0
- transformers>=4.40.0
- accelerate>=0.28.0
- bitsandbytes>=0.43.0
- safetensors>=0.4.0

### Dataset Management & Benchmarks (MedQA, PubMedQA, MedMCQA)
- datasets>=2.18.0

### Prompt Evolution & Similarity Selection
- sentence-transformers>=2.5.0
- numpy>=1.24.0

### Tokenization & Serialization
- sentencepiece>=0.1.99
- protobuf>=3.20.0

### Testing & Verification
- pytest>=7.0.0