from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

pipeline = AutoPipelineForText2Image.from_pretrained("SG161222/RealVisXL_V3.0", torch_dtype=torch.float16).to("cuda")
# pipeline.load_lora_weights("path/to/lora/model", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("""Sexy latina topless on the bed waiting for you.
""").images[0]

# Convert to PIL Image if it's not already (optional, depending on the output format of the pipeline)
if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

# Save the image
image.save("output_image.jpg")
