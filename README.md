# RotorQuant: KV Cache Compression for LLMs

Drop-in KV cache quantization using rotation-based decorrelation. **4-5x compression** with **+19% PPL** degradation at 4-bit. Works with PyTorch/Triton (CUDA) and [llama.cpp (Metal/Apple Silicon)](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache).

## Results

### Perplexity (Qwen2.5-3B, wikitext-2, post-prefill quantization)

| Method | 3-bit PPL | 4-bit PPL | 3-bit vs FP16 | 4-bit vs FP16 |
|--------|-----------|-----------|---------------|---------------|
| **FP16** | **7.59** | **7.59** | baseline | baseline |
| **IsoQuant** | 12.35 | **9.03** | +63% | **+19%** |
| **PlanarQuant** | **10.12** | 9.56 | **+33%** | +26% |
| RotorQuant | 12.22 | 10.03 | +61% | +32% |

Reproduce:
```bash
python -m turboquant.benchmark_google_parity --model Qwen/Qwen2.5-3B-Instruct --bits 3 4
```

### Speed (RTX 5090, Triton fused kernels, d=128, 8192 vectors)

| Kernel | Latency | vs RotorQuant |
|--------|---------|---------------|
| **PlanarQuant** | **30 µs** | **88x faster (PyTorch), tied (Triton)** |
| **IsoQuant-Fast** | **30 µs** | **88x faster (PyTorch), tied (Triton)** |
| RotorQuant | 34 µs | baseline |

PyTorch (no Triton): PlanarQuant 164 µs, IsoQuant 466 µs, RotorQuant 2,649 µs.

### VRAM Savings (3-bit, 4.9x compression)

| Context | FP16 KV | Compressed | Saved |
|---------|---------|------------|-------|
| 8K | 288 MB | 59 MB | **230 MB** |
| 32K | 1,152 MB | 234 MB | **918 MB** |
| 65K | 2,304 MB | 469 MB | **1,835 MB** |

### High-Context Generation (Qwen2.5-3B, 3-bit, RTX 5090)

| Context | VRAM | Needle-in-Haystack |
|---------|------|--------------------|
| 8K | 3.1 GB | FOUND |
| 32K | 5.9 GB | FOUND |
| 65K | 9.6 GB | FOUND |

## Quick Start

```bash
pip install -e .
pip install triton  # optional, for fused GPU kernels
```

```python
from turboquant import IsoQuantMSE, PlanarQuantMSE

# IsoQuant: best quality (quaternion 4D rotation)
iq = IsoQuantMSE(d=128, bits=4, mode='fast', device='cuda')
x_hat, indices = iq(x)

# PlanarQuant: fastest (2D Givens rotation)
pq = PlanarQuantMSE(d=128, bits=3, device='cuda')
x_hat, indices = pq(x)
```

## llama.cpp (Apple Silicon / Metal)

Native KV cache types in our [llama.cpp fork](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache). Supports `iso3`, `planar3`, `iso4`, `planar4` alongside existing `turbo3`/`turbo4`.

```bash
git clone https://github.com/johndpope/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/planarquant-kv-cache
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Benchmark
./build/bin/llama-bench -m model.gguf -ngl 99 -ctk planar3 -ctv f16 -p 512,2048 -n 64

# Perplexity
pip install datasets
python3 -c "from datasets import load_dataset; open('/tmp/wiki.txt','w').write('\n'.join(load_dataset('wikitext','wikitext-2-raw-v1',split='test')['text']))"
./build/bin/llama-perplexity -m model.gguf -f /tmp/wiki.txt -ngl 99 -c 512 --chunks 20 --cache-type-k planar3 --cache-type-v f16
```

**Deferred quantization**: `iso3`/`planar3` K-cache allocates as FP16 during prefill (no error compounding), matching Python post-prefill quality. PPL 9.98 on Mac — identical to FP16 baseline.

| Cache | Decode tok/s | PPL (M4 Mac Mini) |
|-------|-------------|-------------------|
| FP16 | 47.4 | 9.98 |
| planar3 (deferred) | **48.3** | **9.98** |
| iso3 (deferred) | 47.9 | 9.98 |
| turbo3 (roundtrip) | 33.9 | 180.3 |

## How It Works

Rotation-based quantization decorrelates KV cache vectors before scalar quantization. Three rotation primitives, same pipeline:

1. **Normalize** to unit sphere, store norms separately
2. **Rotate** via block-wise transform (decorrelates coordinates)
3. **Quantize** each coordinate to Lloyd-Max optimal centroids
4. **Inverse rotate** to reconstruct

| | Block | FMAs (d=128) | Parameters | Quality |
|---|-------|-------------|------------|---------|
| TurboQuant (Google) | Dense d×d | 16,384 | 16,384 | baseline |
| RotorQuant | 3D Clifford | 2,408 | 344 | 1.0x |
| **IsoQuant** | **4D quaternion** | **512** | **128** | **1.0x** |
| **PlanarQuant** | **2D Givens** | **256** | **128** | **1.0x** |

All methods achieve identical MSE. The rotation only affects speed and parameter count.

**Post-prefill strategy**: Prefill at full FP16 (no quantization → no error compounding), then bulk-quantize the cache before decode. This gives 3x better PPL than roundtrip quantization.

## Benchmarks

```bash
# All-in-one comparison (MSE, PPL, NIAH, speed)
python -m turboquant.benchmark_google_parity

# Roundtrip PPL (all backends)
python -m turboquant.benchmark_perplexity --bits 3 4 --backends isoquant planarquant rotorquant

# Triton kernel speed
python -m turboquant.benchmark_triton

# High-context generation with compressed KV cache
python -m turboquant.poc_high_context --backend planar --bits 3 --max-ctx 65536
```

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — [Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [IsoQuant / PlanarQuant](https://github.com/ParaMind2025/isoquant) — ParaMind2025
- [QJL: 1-Bit Quantized JL Transform](https://arxiv.org/abs/2406.03482)
- [back2matching/turboquant](https://github.com/back2matching/turboquant) — Reference TurboQuant
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp with TurboQuant cache types

## Citation

```bibtex
@article{pope2026rotorquant,
  title={RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression},
  author={Pope, John D.},
  year={2026},
  url={https://github.com/scrya-com/rotorquant}
}
```

## License

MIT
