from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import torch.utils.data as data

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465], img2=True, unif=True):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.img2 = img2
        self.unif = unif
       
    def __call__(self, img, img2=None):

        if self.unif and random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            if self.unif: target_area = random.uniform(self.sl, self.sh) * area
            else: target_area = self.sh * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if not self.img2:
                    if img.size()[0] == 3:
                        img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                        img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                        img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    else:
                        img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                else:
                    img[:, x1:x1+h, y1:y1+w] = img2[:, x1:x1+h, y1:y1+w]
                return img

        return img


class RandomMixing(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
       
    def __call__(self, img, img2=None):
        lam = np.random.beta(self.alpha+1, self.alpha)
        return  lam * img + (1. - lam) * img2
    

class RandomPixels(object):
    def __init__(self, alpha=0.5, unif=True):
        self.alpha = alpha
        self.unif = unif
       
    def __call__(self, img, img2=None):
        if self.unif: alpha = torch.rand(1)[0]*self.alpha
        lam = (torch.rand(img.shape)>alpha).float()
        return  lam * img + (1. - lam) * img2
    

class RandomPixels2(object):
    def __init__(self, alpha=0.5, unif=True, h1=28, h2=14, resample=0):
        self.alpha = alpha
        self.unif = unif

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(h1, resample),
                transforms.ToTensor()
            ])
        self.h2 = h2
       
    def __call__(self, img, img2=None):
        if self.unif: alpha = torch.rand(1)[0]*self.alpha
        else: alpha = self.alpha
        lam = (torch.rand((1, self.h2, self.h2))>alpha).float()
        lam = self.transform(lam)
        return  lam * img + (1. - lam) * img2
  
    
import numpy as np
import random

class MixupDataset(data.Dataset):

    def __init__(self, dataset, fun, mixup_dataset=None):
        self.dataset = dataset
        self.fun = fun
        self.mixup_dataset = mixup_dataset if mixup_dataset else dataset


    def __getitem__(self, index):
        x,y = self.dataset.__getitem__(index)
        x2,_ = self.mixup_dataset[random.randint(0,len(self.mixup_dataset)-1)]
        x = self.fun(x, x2)
        return x,y

    def __len__(self):
            return self.dataset.__len__()

        

class Subset(data.Dataset):

    def __init__(self, dataset, n):
        self.n = n
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.n
