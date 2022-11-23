import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

from torch.utils.data import Dataset
from augmenters import geomatric_augs
from fmutils import fmutils as fmu
from tabulate import tabulate
import cv2
import numpy as np
from data_utils import std_norm



class CycelGANDataset(Dataset):
    def __init__(self, root_dir, img_height, img_width, train=True, normalize=True, augment_data=False):
        if train:
            self.root_dir = root_dir + 'train/'
        else:
            self.root_dir = root_dir + 'test/'
        self.dataAdir, self.dataBdir = fmu.get_all_dirs(self.root_dir)
        self.dataA = fmu.get_all_files(self.dataAdir)
        self.dataB = fmu.get_all_files(self.dataBdir)

        self.dataAlen = len(self.dataA)
        self.dataBlen = len(self.dataB)
        self.datalen = max(self.dataAlen, self.dataBlen)

        self.img_width = img_width
        self.img_height = img_height
        self.augment_data = augment_data
        self.normalize = normalize

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):

        data_sample = {} 

        dataAimg = cv2.imread(self.dataA[index % self.dataAlen])
        dataAimg = cv2.cvtColor(dataAimg, cv2.COLOR_BGR2RGB)
        dataAimg = cv2.resize(
            dataAimg, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        dataBimg = cv2.imread(self.dataB[index % self.dataBlen])
        dataBimg = cv2.cvtColor(dataBimg, cv2.COLOR_BGR2RGB)
        dataBimg = cv2.resize(
            dataBimg, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        if self.normalize:
            dataAimg = std_norm(dataAimg, alpha=-1, beta=1)
            dataBimg = std_norm(dataBimg, alpha=-1, beta=1)

        if self.augment_data:
            dataAimg, dataBimg = geomatric_augs(dataAimg, dataBimg)

        data_sample['imgA'] = dataAimg
        data_sample['imgB'] = dataBimg

        return data_sample
