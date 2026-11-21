"""3D DenseNet brain-age regressor used for fine-tuning."""

import torch.nn as nn
import monai as mn

class MyDenseNet(nn.Module):
    def __init__(self, spatial_dims = 3 , in_channels = 1 , out_channels = 1):
        super(MyDenseNet, self).__init__()
        self.main = mn.networks.nets.DenseNet(3,1,1)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        output = self.main(input)
        output = self.sig(output)
        return output
