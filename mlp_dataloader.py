from torch.utils.data import DataLoader, Dataset
import torch
import os
import cv2
import numpy as np

class SquareDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = []
        self.labels = []
        for label in ["a", "b", "c"]:
            for file in os.listdir(os.path.join(root_dir, label)):
                if file != ".DS_Store":
                    filename = os.path.join(root_dir, label, file)
                    self.images.append(cv2.imread(filename, -1))
                    self.labels.append(label)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]
        label = self.labels[idx]

        center = img.shape[0] // 2
        main_color = img[center, center]
        main_color = main_color.astype(np.float32) / 255.
        mask = np.all(img != 255, axis=-1)
        idxs = np.where(mask)
        y1, y2 = np.min(idxs[0]), np.max(idxs[0])
        x1, x2 = np.min(idxs[1]), np.max(idxs[1])
        width = x2 - x1
        height = y2 - y1
        img_size = img.shape[0]
        # Normalize img_size
        img_size = (img_size - 325.) / (624.-325.)
        sq_size = (width + height) // 2
        # Normalize sq_size
        sq_size = (sq_size-23.) / (159. - 23.)

        sample = main_color.tolist() + [img_size, sq_size]
        sample = np.array(sample, dtype=np.float32)

        if self.transform:
            sample = self.transform(sample)
        if label == "a":
            label = torch.tensor([1, 0, 0], dtype=torch.float32)
        elif label == "b":
            label = torch.tensor([0, 1, 0], dtype=torch.float32)
        elif label == "c":
            label = torch.tensor([0, 0, 1], dtype=torch.float32)
        return sample, label