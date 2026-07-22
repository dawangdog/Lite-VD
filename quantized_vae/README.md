# Quantized 2D VAE Weights

This directory contains the small configuration files needed by the 2D VAE path. The weight file is not included because it exceeds the 50 MB supplement limit.

Expected file after preparation:

```text
quantized_vae/quantized_vae.safetensors
```

The helper script `quantize_vae.py` prepares this file when the required pretrained 2D VAE is available in the environment.
