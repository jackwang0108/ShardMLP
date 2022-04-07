import torch
import torch.nn as nn

import numpy as np

class SharedMLP1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # weight shape: [out_channel, in_channel, ksize]
        # for point-wise shared mlp, the shape is [out_features, in_features, 1]
        self.conv = nn.Conv1d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=1, bias=False
        )

        # set weight the same
        self.dtype = self.conv.weight.dtype
        weight = torch.ones(out_features, in_features)
        for i in range(1, out_features+1):
            weight[i-1] = i
        weight = weight.unsqueeze(dim=-1)
        print("Weight:", weight)
        self.conv.weight = nn.Parameter(weight.to(dtype=self.dtype))
    
    def forward(self, x):
        # in_size: [batch, point_num, in_features]
        # out_size: [batch, point_num, out_features]
        return self.conv(x)



class SharedMLP2D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # weight shape: [out_channel, in_channel, ksize[0], ksize[1]]
        # for point-wise shared mlp, the shape is [out_features, 1, 1, in_features]
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_features,
            kernel_size=(1, in_features), stride=1, padding=0, bias=False
        )

        # set weight to the same
        self.dtype = self.conv.weight.dtype
        weight = torch.ones(out_features, 1, 1, in_features)
        for i in range(1, out_features+1):
            weight[i-1] = i
        print("Weight:", weight)
        self.conv.weight = nn.Parameter(weight.to(dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in_size: [batch, channel, point_num, in_features]
        # out_size: [batch, channel, point_num, out_features]
        x = self.conv(x)
        x = x.permute(0, 3, 2, 1)
        return x





if __name__ == "__main__":
    # There is a lot of implementations of Shared MLP. No matter which api (conv1d, conv2d, Linear, and etc.) is used, as long as 
    # the manipulation (multiply, summation, and etc.) keep the same as the definition, the result is correct.
    # Below are two ways to implement SharedMLP, using either nn.Conv1D or nn.Conv2D
    shared_mlp_1d = SharedMLP1D(in_features=3, out_features=16)
    shared_mlp_2d = SharedMLP2D(in_features=3, out_features=16)

    # 1 example
    points = torch.ones(30, 3, dtype=shared_mlp_1d.dtype)

    result_1d: torch.Tensor = shared_mlp_1d(points.T)
    result_2d: torch.Tensor = shared_mlp_2d(points.unsqueeze(dim=0).unsqueeze(dim=0))

    print(result_1d.shape)
    print(result_2d.shape)

    changed_1d = result_1d.T
    changed_2d = result_2d.squeeze()

    print((changed_1d == changed_2d).all())
