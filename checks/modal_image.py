"""Shared PyTorch environment for manually launched GPU checks and benchmarks."""

import modal

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch==2.9.1", "transformers==4.57.1", "liger-kernel==0.6.4",
                 "safetensors>=0.4.5", "pytest>=8.3,<10", "sentencepiece>=0.2")
)
