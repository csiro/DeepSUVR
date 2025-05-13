# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
#
# This software is released under the terms of the CSIRO Non-Commercial Licence.
# You can find a full copy of the license in the LICENSE.txt file at the root of this project.

from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
     

class SaLUTNetGPUDataset(Dataset):
  
    def __init__(self, csv_file, ref_mask, new_masks, cort_mask, brain_mask, tfm_ds, device):
      
        self.csv_file    = csv_file
        self.ref_mask    = ref_mask
        self.cort_mask   = cort_mask
        self.tfm_ds      = tfm_ds
        self.brain_mask  = brain_mask
        self.new_masks   = new_masks
        self.device      = device
        
        if(new_masks):
            print("Read Tracer specific masks")
            # ref_mask and cort_mask are dictionay of numpy arrays:
            self.ref_mask_gpu       = torch.from_numpy(self.ref_mask).to(device)
            self.cort_mask_gpu      = torch.from_numpy(self.cort_mask).to(device)
        else:
            self.ref_mask_gpu       = torch.from_numpy(self.ref_mask).to(device)
            self.cort_mask_gpu      = torch.from_numpy(self.cort_mask).to(device)
          
        
        self.brain_mask_gpu     = torch.from_numpy(self.brain_mask).to(device)
        
        print("Reading data from {} csv files".format(self.csv_file))
        
        # read meta data
        self.img1_paths, self.sid, self.stp1, self.tr1, self.tfm1  = [], [], [], [], []
        
        df = pd.read_csv(self.csv_file, dtype={'ID': str, 'TP': str})
        
        # Sanity Check: Check for required columns
        required_columns = ['ID', 'TP', 'Tracer', 'Filename']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            exit(1)

        # Validate the Tracer column
        invalid_tracers = df[~df['Tracer'].isin(['PIB','NAV','AV45','FBB','FLUTEMETAMOL'])]
        if not invalid_tracers.empty:
            print(f"Invalid tracers found in the Tracer column:\n{invalid_tracers['Tracer'].unique()}.\nTracer should be one of PIB, NAV, AV45, FBB or FLUTEMETAMOL. ")
            exit(1)

        
        # check that tracer only contain 

        # read data
        for index, row in df.iterrows():

            sid = row.ID
            tp1 = row.TP
            tr1 = row.Tracer
            
            # Extract filename path
            fname_img1 = row.Filename
            
            if not os.path.exists(fname_img1):
                print("PET data does not exit {}".format(fname_img1))
                continue
            
            self.img1_paths.append(fname_img1)
            
            self.sid.append(sid)
            self.stp1.append(tp1)
            
            ## encode the tracer into a 1d array so it can be used as input to the network, and return the transform as well
            tr1,tfm1 = tracer_encoder(tr1)
            self.tr1.append(tr1)
            self.tfm1.append(tfm1)
                
        self.img1_paths  = np.array(self.img1_paths)
        self.tr1 = np.array(self.tr1)
        self.tfm1 = np.array(self.tfm1)
            

    def __len__(self):
        return len(self.img1_paths)
      
    
    def compute_suvr(self, img):
        # SUVR normalise the images
        ref = std_mean(img, self.ref_mask)
                    
        # SUVR normalise the images
        img = img/ref
                
        # Compute suvr
        suvr = std_mean(img, self.cort_mask)
                    
        # normalise the data (assume between 0 and 4)
        img = (img - 2)/2
                    
        # mask the images using the brain mask
        img = img*self.brain_mask
        
        return(img,suvr)


    def compute_suvr_tracer(self, img, tracer):

        # SUVR normalise the images
        ref = std_mean(img, self.ref_mask)
                    
        # SUVR normalise the images
        img = img/ref
                
        # Compute suvr
        suvr = std_mean(img, self.cort_mask)
        
        ## account for new mask's tfm:
        suvr = (suvr - self.tfm_ds.loc[tracer]['Intercept']) / self.tfm_ds.loc[tracer]['Slope']
                    
        # normalise the data (assume between 0 and 4)
        img = (img - 2)/2
                    
        # mask the images using the brain mask
        img = img*self.brain_mask
        
        return(img,suvr)


    def reshape(self, arr, mask):
        # reshape the 1D array into a 3D image using the mask
        img = np.zeros(mask.shape, dtype=np.float32).flatten()

        img[mask.flatten()!=0] = arr[:]
        return img.reshape(mask.shape)
      
      
    def reshape_gpu(self, arr_gpu, mask_gpu):  
      
        new_img_gpu = torch.zeros_like(mask_gpu, dtype=torch.float).view(-1)  # Create flattened zero tensor
        new_img_gpu[mask_gpu.flatten() != 0] = arr_gpu  # Assign masked values
        return new_img_gpu.view(*mask_gpu.shape)  # Reshape to original shape



    def __getitem__(self, idx):
      
        fname_img1 = self.img1_paths[idx]
        
        # read the images
        img1 = sitk.ReadImage(fname_img1)
        
        # convert to numpy
        img1 = sitk.GetArrayFromImage(img1).astype(np.float32)
        
        if(self.new_masks):
            img1, suvr1 = self.compute_suvr_tracer(img1, tracer_decoder(self.tr1[idx]))
        else:
            img1, suvr1 = self.compute_suvr(img1)

        img1_gpu    = torch.from_numpy(img1).to(self.device)
               

        # add a new axis
        img1_gpu = img1_gpu.unsqueeze(0)

        return img1_gpu, suvr1, fname_img1, self.sid[idx], self.stp1[idx], self.tr1[idx], self.tfm1[idx]




def std_mean(img, mask):
    mask = mask.astype(np.float32)
    # Implement standard mean
    return (img*mask).sum()/mask.sum()



def std_mean_gpu(img_gpu, mask_gpu, device="cpu"):

    denom = torch.sum(mask_gpu.view(-1))
    
    num = torch.sum((img_gpu * mask_gpu).view(-1))
    
    mean_values = num/denom
    
    return mean_values


def reader(item, pairs, device):
  
  
    img1, suvr1, fname_img1, sid, stp, tr, tfm = item[0], item[1].to(device), item[2], item[3], item[4], item[5].to(device), item[6].to(device)

    # small work around
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if suvr1.ndim == 1:
        suvr1 = suvr1.unsqueeze(0)
        #fname_img1 = fname_img1.unsqueeze(0)

    return img1, suvr1, fname_img1, sid, stp, tr, tfm 


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def wrapped_loader(data_loader, pairs, device):
  
    for item in data_loader:
        img1, suvr1, fname_img1, sid, stp, tr, tfm = reader(item, pairs, device)
        yield img1, suvr1, fname_img1, sid, stp, tr, tfm
    

# Decoder function
def tracer_decoder(tracer):
    if np.array_equal(tracer, [1, 0, 0, 0, 0]):
        return 'PIB'
    elif np.array_equal(tracer, [0, 1, 0, 0, 0]):
        return 'AV45'
    elif np.array_equal(tracer, [0, 0, 1, 0, 0]):
        return 'FLUTEMETAMOL'
    elif np.array_equal(tracer, [0, 0, 0, 1, 0]):
        return 'NAV'
    elif np.array_equal(tracer, [0, 0, 0, 0, 1]):
        return 'FBB'
 



def tracer_encoder(tracer):
  
    if(tracer == 'PIB'):
        return [1,0,0,0,0],[1.07154446763435,1.01397143454706]
    elif(tracer == 'AV45'):
        return [0,1,0,0,0],[0.553349518015116,1.0428965238214554]
    elif(tracer == 'FLUTEMETAMOL'):
        return [0,0,1,0,0],[0.8228031208003321,0.998758708913128]
    elif(tracer == 'NAV'):
        return [0,0,0,1,0],[1.1062261801045712,1.0246336592391196]
    elif(tracer == 'FBB'):
        return [0,0,0,0,1],[0.6529033029281299,1.0066146243020564]
    else:
        return [None],[None]

