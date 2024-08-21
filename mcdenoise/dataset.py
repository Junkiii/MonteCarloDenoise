# Hendrik Junkawitsch; Saarland University

# The costume dataset for training and testing. 

import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2
from enum import IntEnum

class Aux(IntEnum):
    SIMPLE = 3      # only using the noisy image
    NAN    = 9      # using the albedo and normal images as additional feature images
    NDSN   = 12     # using the diffuse, specular, and normal images as additional feature images

class DataSet(Dataset):
    def __init__(self, features, data="data", crop=True):
        root            = os.getcwd();
        self.path       = os.path.join(root, data)
        self.n_samples  = int(len(os.listdir(self.path)) / 6)
        self.features   = features
        self.crop       = crop

        print("initializing image data set", self.path, "with ", self.n_samples, " training samples")
        print("in_channels = ", self.features)

    def __getitem__(self, idx):
        # Load sample with given index
        sample_tuple    = self.load_sample(idx)
        sample          = []
        
        # Image transformation to valid format
        for i, img in enumerate(sample_tuple):
            img = torch.from_numpy(img)
            img = img.permute(2,0,1)
            img = img / 255
            sample.append(img)
        
        # Data augmentation: random crop to 256x256 images
        if(self.crop): sample = self.random_crop(sample, 256)

        # Creating the final tensor and ground truth image for the network
        # depending on the use of auxiliary feature images
        if self.features == Aux.NAN:
            training = torch.cat((sample[0],sample[1]),0)
            training = torch.cat((training, sample[2]),0)
            gt       = sample[5]
        elif self.features == Aux.NDSN:
            training = torch.cat((sample[0],sample[3]),0)
            training = torch.cat((training, sample[4]),0)
            training = torch.cat((training, sample[2]),0)
            gt       = sample[5]
        else:
            training = sample[0]
            gt       = sample[5]

        return gt, training

    def __len__(self):
        return self.n_samples

    def load_sample(self, idx):
        noisy    = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_noisy.png")),     cv2.COLOR_BGR2RGB))
        gt       = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_gt.png")),        cv2.COLOR_BGR2RGB))
        albedo   = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_albedo.png")),    cv2.COLOR_BGR2RGB))
        normal   = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_normal.png")),    cv2.COLOR_BGR2RGB))
        diffuse  = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_diffuse.png")),   cv2.COLOR_BGR2RGB))
        specular = np.array(cv2.cvtColor(cv2.imread(os.path.join(self.path, str(idx)+"_specular.png")),  cv2.COLOR_BGR2RGB))
        return (noisy, albedo, normal, diffuse, specular, gt)

    def random_crop(self, sample, size):
        i, j, h, w = transforms.RandomCrop.get_params(sample[0], output_size=(size,size))
        for idx, s in enumerate(sample):
            sample[idx] = TF.crop(s, i, j, h, w)
        return sample

