# Computer Vision Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

A collection of pre-trained and custom computer vision models for various tasks like object detection, image segmentation, and classification. This repository aims to provide readily usable models and examples for common computer vision applications.

## ✨ Features

-   **Pre-trained Models:** Integration with popular pre-trained models from TensorFlow Hub and PyTorch Hub.
-   **Custom Model Architectures:** Examples of building custom CNNs for specific tasks.
-   **Data Preprocessing:** Utilities for image loading, augmentation, and transformation.
-   **Evaluation Metrics:** Tools for evaluating model performance (e.g., mAP, IoU, accuracy).
-   **Deployment Examples:** Guidance for deploying computer vision models on various platforms.

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/xxavierlsz/computer-vision-models.git
cd computer-vision-models
pip install -r requirements.txt
```

### Usage

Example of using a pre-trained ResNet for image classification:

```python
# classify_image.py
import torch
from torchvision import models, transforms
from PIL import Image
import os

def main():
    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval() # Set model to evaluation mode

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create a dummy image for demonstration
    dummy_image_path = "dummy_image.jpg"
    if not os.path.exists(dummy_image_path):
        # Create a simple red image
        img = Image.new("RGB", (224, 224), color = (255, 0, 0))
        img.save(dummy_image_path)

    # Load and preprocess the image
    img = Image.open(dummy_image_path)
    img_tensor = preprocess(img)
    batch_t = torch.unsqueeze(img_tensor, 0)

    # Make prediction
    with torch.no_grad():
        out = model(batch_t)

    # Load ImageNet class names (for demonstration, a simplified version)
    # In a real scenario, you would load a comprehensive list
    class_names = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "red image"]
    
    # Get the predicted class
    _, index = torch.max(out, 1)
    predicted_class_idx = index.item()

    # Map to a class name (simplified for dummy image)
    if predicted_class_idx >= len(class_names):
        predicted_class_name = "unknown class"
    else:
        predicted_class_name = class_names[predicted_class_idx]

    print(f"Predicted class: {predicted_class_name}")

if __name__ == \"__main__\":
    main()
```

To run the example:

```bash
python classify_image.py
```

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📞 Contact

Alexander Thorne - [alex.thorne.ai@example.com](mailto:alex.thorne.ai@example.com)
Project Link: [https://github.com/xxavierlsz/computer-vision-models](https://github.com/xxavierlsz/computer-vision-models)
