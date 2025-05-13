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
    parser.add_argument('--out_csv', type=str, required=True, help="Output filename")
    parser.add_argument('--in_csv', type=str, default='stats.csv', required=True, help="csv file of input images")
    parser.add_argument('--cort_mask', type=str, default='Masks/Standard/voi_Cortex_2mm.nii', help="target mask (used to establish the target region where we expect to see a longitudinal increase)")
    parser.add_argument('--ref_mask',  type=str, default='Masks/Standard/voi_WhlCbl_2mm.nii', help="reference mask (used to prenormalise the images")
    parser.add_argument('--brain_mask',  type=str, default='Masks/Standard/brainmask.nii', help="brain mask (used to mask the images so that the network doesn't use information from outside the brain for normalisation")
    parser.add_argument('--batch_size', type=int, default=40, help="batch size")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use")
    parser.add_argument('--checkpoints', nargs='+', type=str, required=True, help="List of checkpoints")

   
    args = parser.parse_args()
    print("Prediction script:\nArgs={}".format(args))

    cort_mask  = sitk.GetArrayFromImage(sitk.ReadImage(args.cort_mask))
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(args.brain_mask))
    ref_mask   = sitk.GetArrayFromImage(sitk.ReadImage(args.ref_mask))
   

    print("Setup Testing dataset")
    test_set = dataGPU.SaLUTNetGPUDataset(args.in_csv, ref_mask, False, cort_mask, brain_mask, None, args.device)
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)    
    print("Test set loaded with {} subjects and {} samples".format(len(test_set),len(test_loader.dataset)))

    
    model = networks.SaLUTNetCL4()

    model = model.to(args.device)
    
    print("Total number of learnable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    num_models = len(args.checkpoints)
    num_samples = len(test_set)
    
    #torch.set_flush_denormal(True)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
      
    scalings  = np.zeros(shape=(len(args.checkpoints),(num_samples)), dtype=np.float32)
    pred_suvr = np.zeros_like(scalings)
    pred_cl   = np.zeros_like(scalings)
    
    print(scalings.shape)

    # run scalings
    for n, check_path in enumerate(args.checkpoints):
    
        with torch.no_grad():         
            # load checkpoint into network
            checkpoint = torch.load(check_path, map_location=torch.device(args.device))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Predicting with {}".format(check_path))

            # accumulate scalings
            scaling_preds, suvr_preds, suvr_orgs, fnames_arr = [], [], [], []
            sids_arr, stps_arr, trs_arr = [], [], []
            cl_preds, cl_orgs = [], []
            
            for test_sample in tqdm.tqdm(test_loader):
              
                imgs, suvrs, fnames, sids, stps, trs, tfm = dataGPU.reader(test_sample,  False, args.device)
                
                scaling_factors = model.forward(imgs, trs) * 2

                scaled_suvrs = suvrs * scaling_factors.view(-1)
            
                # Transform the original suvr into cl
                cls         = 100*(suvrs-tfm[:,1])/tfm[:,0]
                scaled_cls  = 100*(scaled_suvrs-tfm[:,1])/tfm[:,0]
            
                scaling_preds.append(scaling_factors.cpu().numpy().T)
                
                # Get the original SUVRs
                suvr_orgs.append(suvrs.cpu().numpy())
                
                # Get the new SUVRs
                suvr_preds.append(scaled_suvrs.cpu().numpy())
                
                
                # Get the original CLs
                cl_orgs.append(cls.cpu().numpy())
                
                # Get the new SUVRs
                cl_preds.append(scaled_cls.cpu().numpy())
                
                fnames_arr.append(fnames)
                sids_arr.append(sids)
                stps_arr.append(stps)
                
                # Apply decoder function to each row of the array
                trs = trs.cpu().numpy()
                decoded_trs = np.array([dataGPU.tracer_decoder(row) for row in trs])
                
                trs_arr.append(decoded_trs)
                
        
            scalings[n][:]  = np.concatenate(scaling_preds, axis=1).squeeze()
            pred_suvr[n][:] = np.concatenate(suvr_preds,    axis=1).squeeze()
            pred_cl[n][:]   = np.concatenate(cl_preds,    axis=1).squeeze()
            
            if(n==len(args.checkpoints)-1):
                ref_cl    = np.concatenate(cl_orgs,     axis=1).squeeze()
                ref_suvr  = np.concatenate(suvr_orgs,     axis=1).squeeze()
                fnames    = np.concatenate(fnames_arr,  axis=0)
                sids      = np.concatenate(sids_arr,    axis=0)
                stps      = np.concatenate(stps_arr,    axis=0)
                trs       = np.concatenate(trs_arr,     axis=0)
            
    # compute mean accross the 5 models
    scalings  = np.mean(scalings,  axis=0).squeeze()
    pred_suvr = np.mean(pred_suvr, axis=0).squeeze()
    pred_cl   = np.mean(pred_cl,   axis=0).squeeze()
    
    df = pd.DataFrame({'Filename':fnames,'ID':sids,'TP':stps,'Tracer':trs,'SUVR_Standard':ref_suvr,'SUVR_DeepSUVR':pred_suvr, 'CL_Standard':ref_cl,'CL_DeepSUVR':pred_cl, 'Scaling':scalings}, index=range(len(ref_suvr)))

    ofname = args.out_csv
    
    # Extract the folder path
    folder_path = os.path.dirname(ofname)
    
    print(ofname, folder_path)

    # Create the folder if it doesn't exist
    if (folder_path) and (not os.path.exists(folder_path)):
        os.makedirs(folder_path)
    
    df.to_csv(ofname, index=False)
    print("DeepSUVR Centiloids successfully saved to ", ofname)
