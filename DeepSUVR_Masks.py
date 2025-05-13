# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
#
# This software is released under the terms of the CSIRO Non-Commercial Licence.
# You can find a full copy of the license in the LICENSE.txt file at the root of this project.


import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse, os, sys
from model import dataGPU, networks
from datetime import datetime
import pandas as pd
import tqdm
import SimpleITK as sitk
import torch.multiprocessing
    

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser("Ensemble prediction script")
    # DATA LOADING PARAMETERS    
    parser.add_argument('--out_csv',    type=str, required=True, help="Output filename")
    parser.add_argument('--in_csv',     type=str, default='stats.csv', required=True, help="csv file of input images")
    parser.add_argument('--cort_mask',  type=str, default='Masks/DeepSUVR/ds_voi_Cortex_2mm.nii.gz', help="DeepSUVR target mask")
    parser.add_argument('--ref_mask',   type=str, default='Masks/DeepSUVR/ds_voi_Reference_2mm.nii.gz', help="DeepSUVR reference mask")
    parser.add_argument('--tfm',        type=str, default='Masks/DeepSUVR/SUVR_Transform.csv',    help="DeepSUVR Mask transform")
    parser.add_argument('--brain_mask', type=str, default='Masks/Standard/brainmask.nii', help="brain mask (used to mask the images so that the network doesn't use information from outside the brain for normalisation")
    parser.add_argument('--batch_size', type=int, default=40, help="batch size")

   
    args = parser.parse_args()
    print("Prediction script:\nArgs={}".format(args))

    # Read ref/target mask

    cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(args.cort_mask)).astype(np.float32)
    ref_mask  = sitk.GetArrayFromImage(sitk.ReadImage(args.ref_mask)).astype(np.float32)
        
    tfm_ds = pd.read_csv(args.tfm)
    
    tfm_ds.set_index('Tracer', inplace=True)
   
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(args.brain_mask))

    print("Setup Testing dataset")
    test_set = dataGPU.SaLUTNetGPUDataset(args.in_csv, ref_mask, True, cort_mask, brain_mask, tfm_ds, 'cpu')
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)    
    print("Test set loaded with {} subjects and {} samples".format(len(test_set),len(test_loader.dataset)))

    # accumulate scalings
    suvr_orgs, fnames_arr = [], []
    sids_arr, stps_arr, trs_arr = [], [], []
    cl_orgs = []
    
    for test_sample in tqdm.tqdm(test_loader):
      
        imgs, suvrs, fnames, sids, stps, trs, tfm = dataGPU.reader(test_sample,  False, 'cpu')
        
        # Transform the original suvr into cl
        cls         = 100*(suvrs-tfm[:,1])/tfm[:,0]
        
        # Get the original SUVRs
        suvr_orgs.append(suvrs.cpu().numpy())
        
        # Get the original CLs
        cl_orgs.append(cls.cpu().numpy())
        
        fnames_arr.append(fnames)
        sids_arr.append(sids)
        stps_arr.append(stps)
        
        # Apply decoder function to each row of the array
        trs = trs.cpu().numpy()
        decoded_trs = np.array([dataGPU.tracer_decoder(row) for row in trs])
        
        trs_arr.append(decoded_trs)
        

    ref_cl    = np.concatenate(cl_orgs,     axis=1).squeeze()
    ref_suvr  = np.concatenate(suvr_orgs,   axis=1).squeeze()
    fnames    = np.concatenate(fnames_arr,  axis=0)
    sids      = np.concatenate(sids_arr,    axis=0)
    stps      = np.concatenate(stps_arr,    axis=0)
    trs       = np.concatenate(trs_arr,     axis=0)
            
    df = pd.DataFrame({'Filename':fnames,'ID':sids,'TP':stps,'Tracer':trs,'SUVR_DS_Mask':ref_suvr,'CL_DS_Mask':ref_cl}, index=range(len(ref_suvr)))

    ofname = args.out_csv
    
    # Extract the folder path
    folder_path = os.path.dirname(ofname)
    
    print(ofname, folder_path)

    # Create the folder if it doesn't exist
    if (folder_path) and (not os.path.exists(folder_path)):
        os.makedirs(folder_path)
    
    df.to_csv(ofname, index=False)
    print("Centiloids with DeepSUVR Masks successfully saved to ", ofname)
