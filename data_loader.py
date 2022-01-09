import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from PIL import Image, ImageOps


class PoseDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, list_ids, labels,overlap_label, mask_size, added_feature=None, gt_D_mask_info = None):
        
        self.labels = labels
        self.overlap_label = overlap_label
        self.list_ids = list_ids
        self.path = path
        self.mask_size = mask_size
        self.added_feature = added_feature
        self.gt_D_mask_info = gt_D_mask_info
        
    def __len__(self):
        
        return len(self.list_ids)
    
    def __getitem__(self, index):

        ID = self.list_ids[index]

        Z = read_image(self.path  + "/Pix3D/crop_mask/" + ID[4:].split(".")[0]+".png")

        mask3 = transforms.Resize((self.mask_size,self.mask_size))(Z)
        mask_gt = mask3[0].reshape(1,self.mask_size,self.mask_size)
        mask = mask_gt

        Z = read_image(self.path  + "/Pix3D/crop_real_masks/" + ID[4:].split(".")[0]+".png")
        mask3 = transforms.Resize((self.mask_size,self.mask_size))(Z)
        mask_real = mask3[0].reshape(1,self.mask_size,self.mask_size)


        resahding_temp = torch.load(self.path  + '/Pix3D/all_new_features/reshading/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu'))
        normal_temp = torch.load(self.path  + '/Pix3D/all_new_features/normal/'+ID[4:].split(".")[0]+'.pt', map_location=torch.device('cpu')).detach()

        features_output = torch.cat((normal_temp.float(), resahding_temp.float()))

        y =  torch.tensor((self.labels[self.labels.index.str.contains( "crop/"+ID[4:].split(".")[0])]).values[-3:])
        y_over =  torch.tensor((self.overlap_label[self.overlap_label.index.str.contains( "crop/"+ID[4:].split(".")[0])]).values[-3:])
        
        full_path = ""
            

        return (features_output.float(), mask.float()), (y[0][0],y[0][1],y[0][2]), ID.split(".")[0].split("/")[1],ID,y_over[0][0], full_path, mask_real