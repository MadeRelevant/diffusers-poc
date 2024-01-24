import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from blip import blip_itm

# Load the BLIP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = blip_itm(pretrained='base', image_size=384, vit='base')
model.eval()
model = model.to(device)


# Function to process and caption an image
def caption_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)

    return caption[0]


# Directory containing images
image_dir = 'path/to/your/image/directory'

# Process each image in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if os.path.isfile(image_path):
        caption = caption_image(image_path, model)
        print(f"Image: {image_name}, Caption: {caption}")
