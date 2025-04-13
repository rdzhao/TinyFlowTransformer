from diffusers.models import AutoencoderKL

import torch
from torchvision import transforms
import einops
import numpy as np
from PIL import Image


from pathlib import Path
import argparse
import os

def test_vae(data_folder, output_folder, num_samples=3, device="cuda"):
    os.makedirs(output_folder, exist_ok=True)

    model = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", revision="fp16", torch_dtype="auto")
    vae.to(device)
    vae.eval()
    
    image_files = sorted(os.listdir(data_folder))
    
    preprocess_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    for image_file in image_files:
        image_path = str(Path(data_folder) / image_file)
        image = np.array(Image.open(image_path).convert("RGB"))

        image_tensor = preprocess_transforms(image)
        image_tensor = image_tensor.to(device)

        image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")
        enc_output = vae.encode(image_tensor)
        latent_dist = enc_output.latent_dist

        for i in range(num_samples):
            latent = latent_dist.sample()
            dec_output = vae.decode(latent)
            dec_image = dec_output.sample.squeeze()
            dec_image = einops.rearrange(dec_image, "c h w -> h w c")

            dec_image = dec_image.detach().cpu().numpy()
            dec_image = np.clip(dec_image, 0, 1)
            dec_image = np.array(dec_image*255, dtype=np.uint8)
            
            dec_image_pil = Image.fromarray(dec_image)
            name, ext = os.path.splitext(image_file)
            dec_image_pil.save(f"{output_folder}/{name}_{i}{ext}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    test_vae(args.data_folder, args.output_folder)

    