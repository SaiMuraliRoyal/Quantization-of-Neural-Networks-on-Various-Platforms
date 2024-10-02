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

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/nn-quantization-comparison.git
   cd nn-quantization-comparison
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the platform-specific directories for examples and benchmarks.

## Usage

Each platform has its own directory with specific instructions and examples:

- `/pytorch`: PyTorch quantization and deployment examples
- `/tensorflow-lite`: TensorFlow Lite quantization and deployment examples
- `/tensorrt`: TensorRT optimization and deployment examples

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

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for TensorRT
- PyTorch team for PyTorch Mobile
- TensorFlow team for TensorFlow Lite

## Contact

For any queries, please open an issue in the GitHub repository or contact the maintainers directly.

---

We hope this project helps you optimize your neural networks for deployment across various platforms. Happy quantizing!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/30956560/5254a8e6-2068-46ed-a7a9-a8d954c26f0d/Effective-Comparision-of-NN-Analysis.pptx
