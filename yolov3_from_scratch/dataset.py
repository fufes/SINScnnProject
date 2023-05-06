# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:18:48 2023

@author: josep
"""

# import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile # to load image
from torch.utils.data import Dataset, DataLoader
from utils import (
   iou_width_height as iou,
   non_max_suppression as nms,
) 

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir, label_dir,
            anchors,
            image_size=416,
            S=[13,26,52],
            C=20,
            transform=None,
    ):
            self.annotations = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.label_dir = label_dir
            self.transform = transform
            self.S = S
            self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # for all three scales
            self.num_anchors = self.anchors.shape[0]
            self.num_anchors_per_scale = self.num_anchors // 3
            self.C = C
            self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] #[po, x, y, w, h, c]
        
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False] # makes sure that there is an anchor for each of the three scales
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x) # get the cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i # both are between [0, 1]
                    width_cell, height_cell = (
                        width * S, 
                        height * S,
                    ) # both can be > 1
                    
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]                        
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates # 1:5 are [x, y, width, height]
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction
            
            return image, tuple(targets)