# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
#
# This software is released under the terms of the CSIRO Non-Commercial Licence.
# You can find a full copy of the license in the LICENSE.txt file at the root of this project.

import torch
import torch.nn as nn

class SaLUTNetCL4(nn.Module):

    def __init__(self):
        super(SaLUTNetCL4, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True), nn.InstanceNorm3d(16), nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True), nn.InstanceNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, bias=True), nn.InstanceNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0, bias=True), nn.InstanceNorm3d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),nn.Dropout(0.5))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608+5, out_features=128, bias=True), nn.BatchNorm1d(128), nn.Tanh(),
            nn.Dropout(0.5), nn.Linear(in_features=128, out_features=64, bias=True), nn.BatchNorm1d(64), nn.Tanh(),
            nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid()
        )
      
    # Add extra input to define the tracer used
    def forward(self, x, y):
        x = self.layers(x)
        #print(x.shape, y.shape)
        x = self.classifier(torch.concat([x,y],dim=1))
        return x     
      
