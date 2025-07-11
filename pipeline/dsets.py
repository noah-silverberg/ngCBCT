import os
import scipy.io
import numpy as np

import torch
from torch.utils.data import Dataset


def normalizeInputs(input_images):
    # remove the first and last 16 rows and column for range computation
    x_max = 496
    x_min = 16
    # remove the first and last 20 slices
    c_max = 180
    c_min = 20

    range_max = np.max(input_images[x_min:x_max, x_min:x_max, c_min:c_max])
    input_images = input_images[:, :, c_min:c_max]
    input_images *= 1 / range_max

    return input_images


def normalizeInputsClip(input_images):

    # remove the first and last 20 slices
    c_max = 180
    c_min = 20
    input_images = torch.transpose(input_images[c_min:c_max, :, :], 1, 2)

    # clip input images to [0, 0.04]
    input_images.clip_(0, 0.04)

    # normalize input to [0, 1]
    input_images *= 25.0

    return input_images


def data_augment_horizontal(input_images):

    # flip horizontal
    for i in range(input_images.shape[0]):
        input_images[i] = input_images[i].flip(2)
    return input_images


def data_augment_vertical(input_images):

    # flip vertical
    for i in range(input_images.shape[0]):
        input_images[i] = input_images[i].flip(1)
    return input_images


class CTSet(Dataset):
    def __init__(self, base_dir):
        super(CTSet, self).__init__()
        # self.transforms = transforms  # make sure transforms has at leat ToTensor()
        self.data = []  # store all data here
        # go over all files in base_dir
        for file in os.listdir(base_dir):
            if file.endswith("FDK_full.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_full"])
            elif file.endswith("FDK.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK"])
            elif file.endswith("HF_ns.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["reconFDK"])
            elif file.endswith("u_PL.b1.iter200.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_PL"])
            # ROI input files
            elif file.endswith("FDK_ROI_fullView.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_ROI_fullView"])
            elif file.endswith("FDK_ROI.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_ROI"])
            elif file.endswith("u_PL_ROI.b1.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_PL_ROI"])
            elif file.endswith("FF_ROI_ns.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["recon_FDK_ROI"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        train_images = self.data[index]
        # normalize inputs to [0, 1]
        train_images = normalizeInputsClip(train_images)
        train_images = torch.from_numpy(train_images).float()
        # rearrange inputs to [index, row, column]
        train_images = train_images.permute(2, 0, 1)
        # nework expects [index, channel, row, column]
        # CT scans only have 1 channel
        train_images = torch.unsqueeze(train_images, 1)
        return train_images


class AugCTSet(Dataset):
    def __init__(self, base_dir):
        super(AugCTSet, self).__init__()
        # self.transforms = transforms  # make sure transforms has at leat ToTensor()
        self.data = []  # store all data here
        # go over all files in base_dir
        for file in os.listdir(base_dir):
            if file.endswith("FDK_full.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_full"])
            elif file.endswith("FDK.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK"])
            elif file.endswith("HF_ns.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["reconFDK"])
            elif file.endswith("u_PL.b1.iter200.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_PL"])
            # ROI input files
            elif file.endswith("FDK_ROI_fullView.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_ROI_fullView"])
            elif file.endswith("FDK_ROI.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_FDK_ROI"])
            elif file.endswith("u_PL_ROI.b2.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["u_PL_ROI"])
            elif file.endswith("FF_ROI_ns.mat"):
                mat = scipy.io.loadmat(os.path.join(base_dir, file))
                self.data.append(mat["recon_FDK_ROI"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        train_images = self.data[index]
        # normalize inputs to [0, 1]
        train_images = normalizeInputsClip(train_images)
        train_images = torch.from_numpy(train_images).float()
        # rearrange inputs to [index, row, column]
        train_images = train_images.permute(2, 0, 1)
        # nework expects [index, channel, row, column]
        # CT scans only have 1 channel
        train_images = torch.unsqueeze(train_images, 1)
        train_images_aug = torch.cat(
            (train_images, train_images.flip(2), train_images.flip(3))
        )
        return train_images_aug


class TestScan(Dataset):
    def __init__(self, test_file):
        super(TestScan, self).__init__()
        # self.transforms = transforms  # make sure transforms has at leat ToTensor()
        self.data = []  # store all data here
        # go over all files in base_dir
        if test_file.endswith("FDK_full.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_FDK_full"])
        elif test_file.endswith("FDK.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_FDK"])
        elif test_file.endswith("HF_ns.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["reconFDK"])
        elif test_file.endswith("u_PL.b1.iter200.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_PL"])
        # ROI input files
        elif test_file.endswith("FDK_ROI_fullView.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_FDK_ROI_fullView"])
        elif test_file.endswith("FDK_ROI.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_FDK_ROI"])
        elif test_file.endswith("u_PL_ROI.b1.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["u_PL_ROI"])
        elif test_file.endswith("FF_ROI_ns.mat"):
            mat = scipy.io.loadmat(test_file)
            self.data.append(mat["recon_FDK_ROI"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        train_images = self.data[index]
        # normalize inputs to [0, 1]
        train_images = normalizeInputsClip(train_images)
        train_images = torch.from_numpy(train_images).float()
        # rearrange inputs to [index, row, column]
        train_images = train_images.permute(2, 0, 1)
        # nework expects [index, channel, row, column]
        # CT scans only have 1 channel
        train_images = torch.unsqueeze(train_images, 1)
        return train_images


class TrainSet(Dataset):
    def __init__(self, base_dir, mode="train"):
        super(TrainSet, self).__init__()

        data_ver = "data/DS1/"

        self.mode = mode
        if self.mode == "train":
            self.files_A = torch.load(base_dir + data_ver + "train/P01_HF01_full.pt")
            self.files_B = torch.load(base_dir + data_ver + "train/P01_HF01_ns.pt")
        elif self.mode == "validation":
            self.files_A = torch.load(base_dir + data_ver + "train/P01_HF01_full.pt")
            self.files_B = torch.load(base_dir + data_ver + "train/P01_HF01_ns.pt")
        # elif self.mode == 'test':
        #     self.files_A = torch.load(
        #         base_dir + data_ver + 'test/full.pt')
        #     self.files_B = torch.load(base_dir + data_ver + 'test/ns.pt')

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        item_A = self.files_A[index]
        item_B = self.files_B[index]
        item_A = torch.unsqueeze(item_A, 0)
        item_B = torch.unsqueeze(item_B, 0)
        return {"A": item_A, "B": item_B}


# class PrjSet(Dataset):
#     def __init__(self, base_dir, scans, device):
#         super(PrjSet, self).__init__()
#         # self.transforms = transforms  # make sure transforms has at leat ToTensor()
#         self.data = []  # store all data here
#         # go over all files in base_dir
#         for patient, scan, scan_type in scans:
#             self.prj = torch.load(os.path.join(base_dir, f'{scan_type}_p{patient}_{scan}.pt')).to(device).detach()
#             self.data.append(self.prj)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         train_images = self.data[index]
#         return train_images


class PairSet(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)


class PairNumpySet(Dataset):
    def __init__(self, tensor_path_1, tensor_path_2, device, augment_on_fly=False, recon_len=None):
        # Load only metadata (not full tensors)
        self.tensor_1 = np.load(tensor_path_1, mmap_mode="r")
        self.tensor_2 = np.load(tensor_path_2, mmap_mode="r")
        factor = 3 if augment_on_fly else 1
        if self.tensor_1.shape[0] * factor != self.tensor_2.shape[0] or self.tensor_1.shape[1:] != self.tensor_2.shape[1:]:
            raise ValueError("Tensors must have the same shape.")
        self.length = self.tensor_2.shape[0]  # Number of samples
        self.device = device
        self.augment_on_fly = augment_on_fly
        self.recon_len = recon_len

        if self.augment_on_fly and recon_len is None:
            raise ValueError("recon_len must be provided when augment_on_fly is True.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load only the required slice
        if self.augment_on_fly:
            slice_idx = idx % self.recon_len
            vol_num = idx // (3 * self.recon_len)
            if (idx // self.recon_len) % 3 == 0:
                img1 = torch.tensor(self.tensor_1[slice_idx + vol_num * self.recon_len]).to(self.device)
            elif (idx // self.recon_len) % 3 == 1:
                img1 = torch.tensor(self.tensor_1[slice_idx + vol_num * self.recon_len]).flip(1).to(self.device)
            else:
                img1 = torch.tensor(self.tensor_1[slice_idx + vol_num * self.recon_len]).flip(2).to(self.device)
        else:
            img1 = torch.tensor(self.tensor_1[idx]).to(self.device)
        img2 = torch.tensor(self.tensor_2[idx]).to(self.device)
        return img1, img2  # Return as a paired sample