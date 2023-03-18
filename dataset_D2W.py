from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

import config


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        image = np.array(img)
        DWI_image = image[:, 256:, :]  # H W  channels
        FLAIR_image = image[:, :256, :]
        #  数据增强  具体查看config.py
        if self.transform:
            augmentations = self.transform(image=FLAIR_image, image0=DWI_image)
            FLAIR_image = augmentations["image"]
            DWI_image = augmentations["image0"]

        return DWI_image, FLAIR_image


if __name__ == "__main__":
    dataset = MRIDataset("DATA_PNG/train/", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=2)
    for x, y in loader:
        print(x.shape)
