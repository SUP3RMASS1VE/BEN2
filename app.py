import gradio as gr
import torch
import os
import sys
from loadimg import load_img
from ben_base import BEN_Base
import random
import numpy as np
import huggingface_hub
from gradio.themes import Base
from gradio.themes.utils import colors, fonts, sizes

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(9)
torch.set_float32_matmul_precision("high")

model = BEN_Base()

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = os.path.join(models_dir, "BEN2_Base.pth")

# Check if the model file exists locally
if not os.path.exists(model_path):
    # Download the model file from Hugging Face Hub
    model_path = huggingface_hub.hf_hub_download(
        repo_id="PramaLLC/BEN2",
        filename="BEN2_Base.pth",
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model.loadcheckpoints(model_path)
model.to(device)
model.eval()

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    result_image = process(im)
    image_path = os.path.join(output_folder, "foreground.png")
    result_image.save(image_path)
    return result_image, image_path

def process(image):
    foreground = model.inference(image)
    print(type(foreground))
    return foreground

def process_file(f):
    name_path = f.rsplit(".",1)[0]+".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

# Create a custom theme
class ModernTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.indigo,
            secondary_hue=colors.slate,
            neutral_hue=colors.slate,
            font=(fonts.GoogleFont("Inter"), fonts.GoogleFont("IBM Plex Mono")),
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
        )

# Create the Gradio interface with enhanced styling
def create_interface():
    with gr.Blocks(theme=ModernTheme()) as demo:
        gr.Markdown(
            """
            # 🎨 BEN2 Background Remover
            
            A powerful AI model for removing backgrounds from images with high precision.
            
            ### How to use:
            1. Upload an image using the input area below
            2. Wait for processing
            3. Download your result with transparent background
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                )
                submit_btn = gr.Button(
                    "Remove Background",
                    variant="primary",
                    scale=1
                )
            
            with gr.Column():
                output_image = gr.Image(
                    label="Result",
                    type="pil",
                )
                download_btn = gr.File(
                    label="Download PNG",
                    file_count="single"
                )
        
        # Add footer with additional information
        gr.Markdown(
            """
            ### Tips
            - For best results, ensure your image has good lighting and contrast
            - Supported formats: JPG, PNG, WEBP
            - Maximum image size: 4096x4096 pixels
            
            [View Model on Hugging Face](https://huggingface.co/PramaLLC/BEN2) | [Report Issues](https://github.com/yourusername/yourrepo/issues)
            """
        )
        
        submit_btn.click(
            fn=fn,
            inputs=input_image,
            outputs=[output_image, download_btn],
        )
    
    return demo

# Create and launch the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
    )