import os 
import pandas as pd 
import albumentations
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Casia_surf(Dataset):

    def __init__(self, data_dir, csv_file, resize=None):
        self.data_dir = data_dir 
        self.dataframe = pd.read_csv(csv_file)
        self.resize = resize
        self.aug = albumentations.Compose([
            albumentations.Normalize(always_apply=True),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 
                                self.dataframe.iloc[idx, 0])
        label = self.dataframe.iloc[idx, 1]

        image = Image.open(img_path)

        if self.resize is not None: 
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)
        augmented = self.aug(image = image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype = torch.float), 
            "labels": torch.tensor(label, dtype = torch.float)
        }


if __name__ == "__main__":

    data_dir = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data"
    train_csv = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\train_folds.csv"
    dataset = Casia_surf(data_dir, train_csv, (224, 224))
    
    for i in range(len(dataset)):
        sample =  dataset[i]

        print(i, sample["images"].shape, sample["labels"].item())

        if i == 3:
            break