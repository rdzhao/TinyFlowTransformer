from dataset import FlowDataset
from model import FlowModel

from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import einops

import numpy as np
from PIL import Image

import yaml
import argparse
import os

class FlowTrainer():
    def __init__(self, config_path, device):
        self.config_path = config_path
        self.device = device

        with open(self.config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.dataset_config = config["dataset"]
        self.model_config = config["model"]
        self.train_config = config["train"]
        self.eval_config = config["eval"]

        self.dataset = FlowDataset(self.dataset_config["data_folder"])
        self.dataloader = DataLoader(self.dataset, batch_size=self.dataset_config["batch_size"], num_workers=4)

        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="vae", 
            revision="fp16", 
            torch_dtype="auto"
        )
        self.vae.to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_encoder.to(device)
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False

        self.model = FlowModel(
            n_blocks=self.model_config["n_blocks"], 
            c_latent=self.model_config["c_latent"], 
            d_embd=self.model_config["d_embd"], 
            n_head=self.model_config["n_head"], 
            d_time_embd=self.model_config["d_time_embd"], 
            d_cond_embd=self.clip_text_encoder.config.hidden_size, 
            patch_size=self.model_config["patch_size"],
        )
        self.initialize_weights()
        self.model.to(device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.train_config["learning_rate"], 
            weight_decay=self.train_config["weight_decay"]
        )

        self.loss = nn.MSELoss()

    def initialize_weights(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def train_one_step(self, image, desc):
        image = image.to(self.device)
        b, c, h, w = image.shape
        with torch.no_grad():
            image_enc = self.vae.encode(image)
            latent = image_enc.latent_dist.sample()
        l_b, l_c, l_h, l_w = latent.shape

        with torch.no_grad():
            inputs = self.clip_tokenizer(
                desc,
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            cond = self.clip_text_encoder(**inputs).last_hidden_state

        noise = torch.randn(l_b, l_c, l_h, l_w).to(image.device)

        time = torch.rand(b).to(image.device) ** 2
        time = torch.clamp(time, min=1e-4)
        time_broadcast = einops.rearrange(time, "b -> b 1 1 1")
        
        # x = noise + t*(latent - noise) -> x + (1-t)*(latent-noise) = latent 
        x = time_broadcast * latent + (1 - time_broadcast)*noise
        
        vf_pred = self.model(x, time, cond)

        loss = self.loss(latent - noise, vf_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):
        for i in range(self.train_config["n_epochs"]):
            for k, batch in enumerate(self.dataloader):
                loss = self.train_one_step(batch[0], batch[1])
                print("Iteration:", k, "Loss:", loss.detach().cpu().numpy())

                if k % self.eval_config["iters"] == 0:
                    self.eval(k ,batch[0][0:2], batch[1][0:2])

                if k % self.train_config["save_ckpt_interval"] == 0:
                    self.save_checkpoint(k)

    @torch.no_grad()
    def eval(self, num_iters, image, desc):
        image = image.to(self.device)
        b, c, h, w = image.shape
        with torch.no_grad():
            image_enc = self.vae.encode(image)
            latent = image_enc.latent_dist.sample()
        l_b, l_c, l_h, l_w = latent.shape

        with torch.no_grad():
            inputs = self.clip_tokenizer(
                desc,
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            cond = self.clip_text_encoder(**inputs).last_hidden_state

        latent = torch.randn(l_b, l_c, l_h, l_w, device=image.device)
        time_step = torch.tensor([1 / self.eval_config["steps"]], device=image.device)
        for i in range(self.eval_config["steps"]):
            time = torch.tensor([i / self.eval_config["steps"]], device=image.device)
            vf_pred = self.model(latent, time, cond)
            latent += vf_pred * time_step

        dec_outputs = self.vae.decode(latent)
        dec_images = dec_outputs.sample
        dec_images = einops.rearrange(dec_images, "b c h w -> b h w c")

        dec_images = dec_images.detach().cpu().numpy()
        dec_images = np.clip(dec_images, 0, 1)
        dec_images = np.array(dec_images*255, dtype=np.uint8)
        
        eval_folder = self.eval_config["eval_folder"]
        os.makedirs(eval_folder, exist_ok=True)
        for i in range(len(dec_images)):
            dec_image_pil = Image.fromarray(dec_images[i])
            iter_folder = f"{eval_folder}/iteration_{num_iters}"
            os.makedirs(iter_folder, exist_ok=True)
            dec_image_pil.save(f"{iter_folder}/img_{i}.jpg")

    def save_checkpoint(self, iterations):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        ckpt_folder = self.train_config["ckpt_folder"]
        os.makedirs(ckpt_folder, exist_ok=True)

        filepath = f"{ckpt_folder}/iteration_{iterations}.pth"
        torch.save(checkpoint, filepath)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    device = "cuda"

    trainer = FlowTrainer(args.config_path, device)
    trainer.train()
        