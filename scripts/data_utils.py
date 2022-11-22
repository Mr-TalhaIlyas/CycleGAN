import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
from torchvision import transforms
import cv2
import numpy as np

transformer = transforms.Compose([
                                 # this transfrom converts BHWC -> BCHW and 
                                 # also divides the image by 255 by default if values are in range 0..255.
                                 transforms.ToTensor(),
                                ])

normalize = lambda x, alpha, beta : (((beta-alpha) * (x-np.min(x))) / (np.max(x)-np.min(x))) + alpha
standardize = lambda x : (x - np.mean(x)) / np.std(x)

def std_norm(img, norm=True, alpha=0, beta=1):
    '''
    Standardize and Normalizae data sample wise
    alpha -> -1 or 0 lower bound
    beta -> 1 upper bound
    '''
    img = standardize(img)
    if norm:
        img = normalize(img, alpha, beta) 
    return img

def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    
    imgA = [ b['imgA'] for b in batch ] # w, h
    imgB = [ b['imgB'] for b in batch ]
   
    return {'imgA': imgA, 'imgB': imgB}

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs