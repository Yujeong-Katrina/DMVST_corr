import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialView(nn.Module):
    def __init__(self, city, S=9, num_layers=3, num_filters=64, filter=3, d_out=64):
        super(SpatialView, self).__init__()
        
        self.d = d_out
        self.lamda = num_filters
        self.filter_size = filter
        self.S = S  

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.lamda, kernel_size=self.filter_size, padding=1),
            nn.BatchNorm2d(self.lamda),
            nn.ReLU(),
            nn.Conv2d(self.lamda, self.lamda, kernel_size=self.filter_size, padding=1),
            nn.BatchNorm2d(self.lamda),
            nn.ReLU(),
            nn.Conv2d(self.lamda, self.lamda, kernel_size=self.filter_size, padding=1),
            nn.BatchNorm2d(self.lamda),
            nn.ReLU()
        )

        padding_size = self.S // 2
        self.final_conv = nn.Conv2d(self.lamda, self.d, kernel_size=self.S, padding=padding_size)
        self.relu = nn.ReLU()

    def forward(self, city_map):

        feat_map = self.conv_layers(city_map)

        out_map = self.relu(self.final_conv(feat_map))
        
        b, c, h, w = out_map.size()

        out = out_map.permute(0, 2, 3, 1).contiguous()
        spatial_features = out.view(b, h * w, c)
        
        return spatial_features