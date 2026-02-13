# Project Roadmap

## 1. Short Term (Immediate)
- [x] **Full Vision Integration**: Implement CLIP/Phi-3 loader.
- [x] **INT4 Quantization**: Deploy 4-bit optimized models.
- [ ] **Flash Attention**: Enable by default on supported GPUs.

## 2. Medium Term (Next Major Release)
- [ ] **Model Upgrade**: Evaluate **Phi-1.5** or **TinyLlama-1.1B** as improved draft models.
- [ ] **Continuous Benchmarking**: Integrate `collect_benchmarks.py` into CI/CD pipeline.
- [ ] **Streaming support**: Implement token-by-token streaming to UE5 to reduce perceived latency.

## 3. Long Term (Future)
- [ ] **Production Deployment**:
    - Packaged installer for non-technical users.
    - Automatic model downloading/updating.
- [ ] **Multi-Agent Coordination**: Enable multiple NPCs to communicate directly.
- [ ] **Voice Integration**: Text-to-Speech (TTS) and Speech-to-Text (STT) modules.
