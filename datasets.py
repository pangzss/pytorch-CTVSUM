import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import h5py
import numpy as np
import json
from utils import *
from random import shuffle

class DPLDatasetRand(Dataset):
    def __init__(self, cfg, mode):
        assert mode in ['train','test']
        assert cfg.setting in ['Augmented','Transfer','Canonical']

        self.mode = mode
        self.num_frames = cfg.num_frames
        self.train_data = []
        self.test_data = []
        dataset = cfg.name

        data_folder = cfg.paths.interim
        
        self.data = {}
        names = ['tvsum','summe','youtube','ovp']
        for n in names:
            self.data[n] = h5py.File(data_folder+'/eccv16_dataset_{}_google_pool5.h5'.format(n), 'r')

        with open(data_folder+'/splits/{}_splits.json'.format(dataset), 'r') as fp:
            splits = json.load(fp)

        test_videos = splits[cfg.split]['test_keys']
        train_videos = splits[cfg.split]['train_keys']

        if cfg.setting == 'Augmented':
            for video in test_videos:
                self.test_data.append(self.data[dataset][video])

            for n in names:
                if n == dataset:
                    for video in train_videos:
                        self.train_data.append(self.data[n][video]) 
                else:
                    for video in list(self.data[n].keys()):
                        self.train_data.append(self.data[n][video])
        elif cfg.setting == 'Transfer':
            for video in self.data[dataset].keys():
                self.test_data.append(self.data[dataset][video])
            
            for n in names:
                if n != dataset:
                    for video in list(self.data[n].keys()):
                        self.train_data.append(self.data[n][video])
        else:
            for video in test_videos:
                self.test_data.append(self.data[dataset][video])
            for video in train_videos:
                self.train_data.append(self.data[dataset][video]) 

    def __len__(self):
        if self.mode == 'train':
            self.len = len(self.train_data)
        else:
            self.len = len(self.test_data)
        return self.len
    
    def __getitem__(self, index):
        if self.mode == 'train':
            feats = torch.Tensor(self.train_data[index]['features'][...])
            length = len(feats)
            if length >= self.num_frames:
                ids = torch.randperm(length)[:self.num_frames]
                ids = torch.sort(ids)[0]
            else:
                ids = torch.arange(length).view(1,1,-1).float()
                ids = F.interpolate(ids,size=self.num_frames, mode='nearest').long().flatten()
            return feats[ids]
        else:
            video = self.test_data[index]
            return video

class Youtube8M(Dataset):
    def __init__(self, cfg, num_frames):
        self.num_frames = num_frames

        self.dirpath = cfg.paths.youtube8M
        self.fname = os.listdir(self.dirpath)
    def dequantize(self,feat_vector, max_quantized_value=2, min_quantized_value=-2):
        ''' Dequantize the feature from the byte format to the float format. '''
        assert max_quantized_value > min_quantized_value
        quantized_range = max_quantized_value - min_quantized_value
        scalar = quantized_range / 255.0
        bias = (quantized_range / 512.0) + min_quantized_value
        return feat_vector * scalar + bias

    def __len__(self):
        return len(self.fname)
    def __getitem__(self, index):
        fp = os.path.join(self.dirpath, self.fname[index])
        feature = np.load(fp)
        deq_feature = torch.tensor(self.dequantize(feature)).float()
        length = len(deq_feature)
        if length >= self.num_frames:
            ids = torch.randperm(length)[:self.num_frames]
            ids = torch.sort(ids)[0]
        else:
            ids = torch.arange(length).view(1,1,-1).float()
            ids = F.interpolate(ids,size=self.num_frames, mode='nearest').long().flatten()
        ret_features = deq_feature[ids]
        ret_features = F.normalize(ret_features,p=2, dim=1)
        return ret_features