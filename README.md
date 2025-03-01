# 🎨 BEN2 Background Remover

## Description
BEN2 Background Remover is an AI-powered tool designed to remove backgrounds from images with high precision. It utilizes the BEN_Base model for processing images and is implemented using Gradio for an interactive web-based interface.

## Features
- Removes backgrounds from images efficiently
- Supports multiple image formats (JPG, PNG, WEBP)
- Processes images up to 4096x4096 pixels
- Simple and intuitive UI using Gradio
- Runs on both CPU and GPU

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- PyTorch (with CUDA if using GPU)

### Install Dependencies
```bash
pip install -r requirements.txt
```
## Usage
### Running the Application
```bash
python app.py
```
This will start a local Gradio server, accessible at `http://127.0.0.1:7860/`.

### How to Use
1. Upload an image using the input area.
2. Click the "Remove Background" button.
3. Wait for the processing to complete.
4. Download the result with a transparent background.

## Model Details
The BEN_Base model is loaded from Hugging Face:
- Repository: [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
- Model File: `BEN2_Base.pth`

If the model file is not present, it will be automatically downloaded.

## Development
### Setting Random Seeds
To ensure reproducibility, random seeds are set for:
- Python’s `random` module
- NumPy
- PyTorch (CPU & CUDA)

### Processing Pipeline
1. Load the image and convert it to RGB.
2. Pass the image through the BEN_Base model for inference.
3. Save and return the processed transparent image.

## UI Design
The interface is built with Gradio using a custom modern theme:
- Primary Hue: Indigo
- Secondary Hue: Slate
- Font: Inter, IBM Plex Mono

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Hugging Face Hub](https://huggingface.co/)
- [Gradio](https://gradio.app/)
- [PyTorch](https://pytorch.org/)

