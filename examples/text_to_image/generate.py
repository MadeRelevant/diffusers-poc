from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
from PIL import Image
import sys
from datetime import datetime

# Check if a prompt is provided via command line arguments
if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "RAW photo, aifluencer, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"

generator = torch.Generator(device="cuda").manual_seed(1337)

pipeline = (AutoPipelineForText2Image.from_pretrained("SG161222/RealVisXL_V3.0", torch_dtype=torch.float16))
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")

pipeline.load_lora_weights("output", weight_name="pytorch_lora_weights.safetensors", adapter_name="aifluencer")
pipeline.fuse_lora(lora_scale=0.8)

image = pipeline(prompt,
                 num_inference_steps=30,
                 guidance_scale=7.5,
                 generator=generator,
                 negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                 # Weight for lora
                 cross_attention_kwargs={"scale": 0.5}
                 ).images[0]

# Convert to PIL Image if it's not already (optional, depending on the output format of the pipeline)
if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

# Generate a unique filename based on the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
filename = f"output/output_image_{timestamp}.jpg"

# Save the image
image.save(filename)
