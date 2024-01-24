from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
from PIL import Image
import sys
from datetime import datetime

# Check if a prompt is provided via command line arguments
if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "photo-of-lies a photo realistic picture taken from front perspective walking on the beach"

generator = torch.Generator(device="cuda").manual_seed(1337)

pipeline = (AutoPipelineForText2Image.from_pretrained("SG161222/RealVisXL_V3.0", torch_dtype=torch.float16))
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.generator = generator
pipeline = pipeline.to("cuda")

pipeline.load_lora_weights("output", weight_name="pytorch_lora_weights.safetensors")
image = pipeline(prompt,
                 num_inference_steps=30,
                 guidance_scale=7.5,
                 negative_prompt="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
                 ).images[0]

# Convert to PIL Image if it's not already (optional, depending on the output format of the pipeline)
if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

# Generate a unique filename based on the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
filename = f"output/output_image_{timestamp}.jpg"

# Save the image
image.save(filename)
