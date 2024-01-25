import json
import os

# Directory containing your image files
image_directory = "data"  # Update this to your directory path

# The caption to be used for all images
# TODO make caption with BLIP or some AI that can extra the caption from the image
caption = "aifluencer doing a selfie pose"

# List all image files in the directory
image_filenames = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]

# Create metadata entries
metadata_entries = []
for filename in image_filenames:
    metadata_entry = {"file_name": filename, "caption": caption}
    metadata_entries.append(metadata_entry)

# Write to metadata.jsonl file
metadata_file_path = os.path.join(image_directory, "metadata.jsonl")
with open(metadata_file_path, "w") as file:
    for entry in metadata_entries:
        json_line = json.dumps(entry)
        file.write(json_line + "\n")

print(f"metadata.jsonl file created successfully in {image_directory}.")
