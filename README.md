# PoorMansDeepSeek

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-blue)](https://huggingface.co/yourname/PoorMansDeepSeek)

A lightweight GPT implementation with **rotary embeddings**, **MoE layers**, and **sparse attention**, built on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). Designed for efficiency on low-resource hardware while maintaining competitive performance with GPT-2-124M.

**Key Additions** ✨
- ✅ Rotary positional embeddings (RoPE) for better long-context modeling
- ✅ 4-bit quantized inference support
- ✅ Mixture-of-Experts (MoE) layer implementation
- 🚀 40% fewer parameters than GPT-2-124M with similar perplexity

---

## Quick Start

### Installation
```bash
git clone https://github.com/yourname/PoorMansNanoGPT
cd PoorMansNanoGPT
pip install -r requirements.txt