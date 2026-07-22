# Quantized 3D VAE Weights

This directory contains the small configuration file needed by the 3D VAE path. The weight file is not included because it exceeds the 50 MB supplement limit.

Expected file after preparation:

```text
quantized_3dvae/quantized_3dvae.pt
```

The helper script `dquantize_3dvae.py` prepares this file when the required pretrained 3D VAE is available in the environment.
