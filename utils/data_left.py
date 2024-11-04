# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
Modified for PyTorch conversion by ChatGPT at 10/15/2024
"""

# Import Libraries:
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class VisualOdometryDataLoaderLeft(Dataset):
    def __init__(self, datapath, height, width, test=False, val=False, sequence_test='05'):
        self.base_path = datapath
        if test or val:
            self.sequences = [sequence_test]
        else:
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            self.sequences = ['00', '02', '08', '09']

        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()
        self.width = width
        self.height = height

        self.images_stacked, self.odometries = self.get_data()

        # Define transformations for the images
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images_stacked)

    def __getitem__(self, idx):
        img1_path, img2_path = self.images_stacked[idx]
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        img = torch.cat((img1, img2), dim=0).float()
        odometry = torch.tensor(self.odometries[idx]).float()
        return img, odometry

    def load_image(self, filepath):
        image = Image.open(filepath)
        if self.transform:
            image = self.transform(image)
        return image

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    def get_image_paths(self, sequence, index):
        image_path = os.path.join(self.base_path, 'KITTI_sequences', sequence, 'image_2',  f'{index:06}.png')
        return image_path

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])
        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])
        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def matrix_rt(self, p):
        return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

    def get_data(self):
        images_paths = []
        odometries = []
        for index, sequence in enumerate(self.sequences):
            for i in range(self.sizes[index] - 1):
                images_paths.append([self.get_image_paths(sequence, i), self.get_image_paths(sequence, i + 1)])
                pose1 = self.matrix_rt(self.poses[index][i])
                pose2 = self.matrix_rt(self.poses[index][i + 1])
                pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                R = pose2wrt1[0:3, 0:3]
                t = pose2wrt1[0:3, 3]
                angles = self.rotationMatrixToEulerAngles(R)
                odometries.append(np.concatenate((t, angles)))
        return np.array(images_paths), np.array(odometries)


def main():
    path = "/home/senselab/KITTI"
    dataset = VisualOdometryDataLoaderLeft(path, 192, 640)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for imgs, odometries in dataloader:
        print(imgs.shape)  # Expecting (batch_size, 6, height, width)
        print(odometries.shape)  # Expecting (batch_size, 6)


if __name__ == "__main__":
    main()
