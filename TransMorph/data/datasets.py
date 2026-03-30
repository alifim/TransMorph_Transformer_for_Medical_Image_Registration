import torch
from torch.utils.data import Dataset
from .data_utils import pkload

import numpy as np
import pandas as pd
import nibabel as nib


class CSVPairDatasetwithTransform(Dataset):
    def __init__(self, csv_path, mode="train", transforms=None):
        """
        Reads your paired CSV.
        mode: 'train' or 'val'.
        transforms: Optional transform functions (e.g., RandomFlip).
        """
        self.df = pd.read_csv(csv_path)

        # Filter by 'train' column if it exists
        if "train" in self.df.columns:
            if mode == "train":
                self.df = self.df[self.df["train"] == True]
            else:
                self.df = self.df[self.df["train"] == False]

        self.df = self.df.reset_index(drop=True)
        self.transforms = transforms
        print(f"Dataset ({mode}): {len(self.df)} pairs loaded.")

    def __len__(self):
        return len(self.df)

    def load_nifti_numpy(self, path):
        if pd.isna(path) or path == "None":
            return None

        img = nib.load(path).get_fdata()

        # Add channel dim: (H, W, D) -> (1, H, W, D)
        if img.ndim == 3:
            img = img[None, ...]

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Images as Numpy Arrays
        moving = self.load_nifti_numpy(row["moving_img_path"])
        fixed = self.load_nifti_numpy(row["fixed_img_path"])
        moving_seg = self.load_nifti_numpy(row.get("moving_seg_path"))
        fixed_seg = self.load_nifti_numpy(row.get("fixed_seg_path"))

        # Handle missing segs (create zeros)
        if moving_seg is None:
            moving_seg = np.zeros_like(moving)
        if fixed_seg is None:
            fixed_seg = np.zeros_like(fixed)

        # Apply Transforms (e.g., RandomFlip)
        if self.transforms is not None:
            transformed = self.transforms([moving, fixed, moving_seg, fixed_seg])
            moving, fixed, moving_seg, fixed_seg = transformed

        # Explicit Type Casting
        moving = moving.astype(np.float32)
        fixed = fixed.astype(np.float32)
        moving_seg = moving_seg.astype(np.int16)
        fixed_seg = fixed_seg.astype(np.int16)

        # Convert to Contiguous Arrays & PyTorch Tensors
        moving = torch.from_numpy(np.ascontiguousarray(moving))
        fixed = torch.from_numpy(np.ascontiguousarray(fixed))
        moving_seg = torch.from_numpy(np.ascontiguousarray(moving_seg))
        fixed_seg = torch.from_numpy(np.ascontiguousarray(fixed_seg))

        return moving, fixed, moving_seg, fixed_seg


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        # print(x.shape)
        # print(x.shape)
        # print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, y = self.transforms([x, y])
        # y = self.one_hot(y, 2)
        # print(y.shape)
        # sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        # print(x.shape)
        # print(x.shape)
        # print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        # y = self.one_hot(y, 2)
        # print(y.shape)
        # sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_seg),
            torch.from_numpy(y_seg),
        )
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
