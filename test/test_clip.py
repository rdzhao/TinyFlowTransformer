from transformers import CLIPTokenizer, CLIPTextModel

import torch

import argparse

from pathlib import Path
import os

def test_clip(data_folder):
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    image_files = sorted(os.listdir(data_folder)) 
    for image_file in image_files:
        text = image_file.split('.')[0]
        inputs = clip_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = clip_text_encoder(**inputs)
        text_embd = outputs.last_hidden_state
        print(text_embd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    args = parser.parse_args()
    
    test_clip(args.data_folder)