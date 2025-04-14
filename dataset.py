import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image

from pathlib import Path
import os

class FlowDataset(IterableDataset):
    def __init__(self, data_folder):
        super().__init__()
        self.data_folder = data_folder
        self.image_files = sorted(os.listdir(self.data_folder))
        self.transforms = transforms.ToTensor()

    def __iter__(self):
        for image_file in self.image_files:
            image_name = image_file.split(".")[0]
            
            image = Image.open(str(Path(self.data_folder) / image_file))
            image = self.transforms(image)

            yield image, image_name