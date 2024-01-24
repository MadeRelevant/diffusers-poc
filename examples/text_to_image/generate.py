from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import sys
from datetime import datetime

# Check if a prompt is provided via command line arguments
if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "At a cocktail bar having a drink in a beautiful dress, half bodyshot"

pipeline = AutoPipelineForText2Image.from_pretrained("SG161222/RealVisXL_V3.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("output", weight_name="pytorch_lora_weights.safetensors")
image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# Convert to PIL Image if it's not already (optional, depending on the output format of the pipeline)
if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

# Generate a unique filename based on the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
filename = f"output_image_{timestamp}.jpg"

# Save the image
image.save(filename)