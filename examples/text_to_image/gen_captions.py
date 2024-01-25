import json
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")


# Directory containing your image files
image_directory = "data"  # Update this to your directory path

# List all image files in the directory
image_filenames = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]

# Create metadata entries
metadata_entries = []
for filename in image_filenames:
    # conditional image captioning
    text = "a photo of"
    inputs = processor(Image.open(os.path.join(image_directory, filename)), text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    metadata_entry = {"file_name": filename, "caption": "aifluencer taking " + caption}
    metadata_entries.append(metadata_entry)

# Write to metadata.jsonl file
metadata_file_path = os.path.join(image_directory, "metadata.jsonl")
with open(metadata_file_path, "w") as file:
    for entry in metadata_entries:
        json_line = json.dumps(entry)
        file.write(json_line + "\n")

print(f"metadata.jsonl file created successfully in {image_directory}.")
