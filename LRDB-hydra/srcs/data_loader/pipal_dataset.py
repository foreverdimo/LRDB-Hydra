import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms
import cv2 as cv
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PIPAL_Dataset(torch.utils.data.Dataset):
    def __init__(self, dst_dir, ref_dir, label_dir, ref_list):
        # ref_list: list of index of used ref images, for split validation
        self.dst_images = []
        self.ref_images = []
        self.scores = []
        self.epsilon = 1
        # self.transform = transforms.Compose(
        #     transforms.Resize((18,18)),
        #     transforms.Resize((288,288)),
        #     transforms.ToTensor()
        # )

        # load images to memory
        print("loading images to memory")
        t = transforms.ToTensor()
        l = os.listdir(label_dir)
        l.sort()
        label_list = []
        for i in ref_list:
            label_list.append(l[i])

        for label_file in tqdm(label_list):
            ref_name = label_file[:-4]+'.bmp'
            ref_image = cv.imread(os.path.join(ref_dir, ref_name),cv.IMREAD_GRAYSCALE)
            self.ref_images.append(t(ref_image))
            with open(os.path.join(label_dir, label_file), 'r') as f:
                labels = f.readlines()
            assert len(labels) == 116
            for item in labels:
                item = item.strip('\n')
                dst_name = item.split(',')[0]
                dst_image = cv.imread(os.path.join(dst_dir, dst_name),cv.IMREAD_GRAYSCALE)
                self.dst_images.append(t(dst_image))
                score = torch.tensor([float(item.split(',')[1])]) / 1400 # score divided by 1400 as groundtruth
                self.scores.append(score)
        
        print("Num of distorted images: "+str(len(self.dst_images)))

    def __len__(self):
        return len(self.dst_images)
    
    def __getitem__(self, idx):
        dist1 = transforms.ToPILImage()(self.dst_images[idx])
        dist = transforms.Resize((18,18))(dist1)
        dist = transforms.Resize((288,288))(dist)
        dist = transforms.ToTensor()(dist)
        refr1 = transforms.ToPILImage()(self.ref_images[idx//116])
        refr = transforms.Resize((18,18))(refr1)
        refr = transforms.Resize((288,288))(refr)
        refr = transforms.ToTensor()(refr)
        dist = dist - transforms.ToTensor()(dist1)
        refr = refr - transforms.ToTensor()(refr1)
        E = (1/np.log2(255**2/self.epsilon)) * torch.log2(1/((dist-refr)**2 + (self.epsilon/255**2)))
        return (dist , E), self.scores[idx]


class PIPAL_Paired_Dataset(torch.utils.data.Dataset):
    def __init__(self, dst_dir, ref_dir, label_dir, ref_list, pairs_per_ref=500):
        # ref_list: list of index of used ref images, for split validation
        self.dst_images = []
        self.ref_images = []
        self.scores = []
        self.pairs_per_ref = pairs_per_ref

        # load images to memory
        print("loading images to memory")
        t = transforms.ToTensor()
        l = os.listdir(label_dir)
        l.sort()
        label_list = []
        for i in ref_list:
            label_list.append(l[i])
            
        for label_file in tqdm(label_list):
            ref_name = label_file[:-4]+'.bmp'
            ref_image = Image.open(os.path.join(ref_dir, ref_name))
            self.ref_images.append(t(ref_image))
            with open(os.path.join(label_dir, label_file), 'r') as f:
                labels = f.readlines()
            assert len(labels) == 116
            for item in labels:
                item = item.strip('\n')
                dst_name = item.split(',')[0]
                dst_image = Image.open(os.path.join(dst_dir, dst_name))
                self.dst_images.append(t(dst_image))
                score = torch.tensor([float(item.split(',')[1])]) / 1400
                self.scores.append(score)
        
        print("Num of reference images: "+str(len(self.ref_images)))
        print("Num of distorted images: "+str(len(self.dst_images)))
        print("Num of image pairs: "+str(len(self.ref_images)*self.pairs_per_ref))

    def __len__(self):
        return len(self.ref_images)*self.pairs_per_ref

    def __getitem__(self, idx):
        ref_idx = idx//self.pairs_per_ref
        ref_image = self.ref_images[ref_idx]
        temp = list(range(116))
        random.shuffle(temp)
        dst1_idx = ref_idx*116+temp[0]
        dst2_idx = ref_idx*116+temp[1]
        return (self.dst_images[dst1_idx], self.dst_images[dst2_idx], ref_image), (self.scores[dst1_idx], self.scores[dst2_idx])


class PIPAL_Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, dst_dir, ref_dir):
        self.dst_dir = dst_dir
        self.ref_dir = ref_dir
        self.dst_names = os.listdir(dst_dir)
        self.dst_names.sort()
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.dst_names)

    def __getitem__(self, idx):
        dst_name = self.dst_names[idx]
        ref_name = dst_name.split('_')[0]+'.bmp'
        dst_image = self.t(Image.open(os.path.join(self.dst_dir, dst_name)))
        ref_image = self.t(Image.open(os.path.join(self.ref_dir, ref_name)))
        return dst_image, ref_image, dst_name
        