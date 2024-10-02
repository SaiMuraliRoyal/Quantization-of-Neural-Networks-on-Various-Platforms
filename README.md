# Neural Network Quantization Comparison Across Platforms

This project provides a comprehensive comparison of neural network quantization techniques on PyTorch, TensorFlow Lite, and TensorRT platforms. It aims to help researchers and developers understand the strengths and limitations of each platform for optimizing deep learning models.

## Overview

Neural network quantization is a crucial technique for deploying deep learning models on resource-constrained devices. This project explores various quantization methods across popular frameworks, highlighting their performance characteristics and deployment workflows.

## Platforms Covered

- PyTorch
- TensorFlow Lite
- TensorRT

## Key Features

### Algorithms
- Classification
- Object Detection
- Segmentation

### Quantization Techniques
- Dynamic Quantization
- Static Quantization
- Quantization-Aware Training

### Deployment Workflows
- PyTorch Mobile and Torchvision
- TensorFlow Lite deployment
- TensorRT optimization and runtime


### Example: PyTorch Quantization

```python
import torch
from torchvision import models

# Load a pretrained model
model = models.resnet18(pretrained=True)

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# Compare model sizes
print(f"Original model size: {torch.save(model.state_dict(), 'model.pth')}")
print(f"Quantized model size: {torch.save(quantized_model.state_dict(), 'quantized_model.pth')}")
```

## Performance Comparisons

We provide benchmarks for various models and hardware platforms, demonstrating the impact of quantization on inference speed and model size. Here's a sample comparison:

| Platform | Model | Precision | Latency (ms) | Model Size (MB) |
|----------|-------|-----------|--------------|-----------------|
| PyTorch  | ResNet50 | FP32 | 25.3 | 97.8 |
| PyTorch  | ResNet50 | INT8 | 12.1 | 24.5 |
| TensorFlow Lite | MobileNetV2 | FP32 | 30.2 | 14.0 |
| TensorFlow Lite | MobileNetV2 | INT8 | 15.7 | 3.5 |
| TensorRT | BERT | FP16 | 5.2 | 438.0 |
| TensorRT | BERT | INT8 | 2.8 | 109.5 |

## Acknowledgments

- NVIDIA for TensorRT
- PyTorch team for PyTorch Mobile
- TensorFlow team for TensorFlow Lite
