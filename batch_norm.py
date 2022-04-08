import torch
import torch.nn as nn

import numpy as np


class SharedMLP1D(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.batch_norm = nn.BatchNorm1d(in_features, eps=0, momentum=0, affine=False, track_running_stats=False)

        self.dtype = torch.float32


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [batch, channel] or [batch, channel, length]
        # for point cloud: [batch, feature_num] or [batch, feature_num, point_num]
        # output shape: the same as input shape
        return self.batch_norm(x)



class SharedMLP2D(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.batch_norm = nn.BatchNorm2d(in_features, eps=0, momentum=0, affine=False, track_running_stats=False)

        self.dtype = torch.float
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [batch, channel, width, height]
        # for point cloud: [batch, 1, point_num, feature_num]
        # output shape: same as input
        return self.batch_norm(x)


if __name__ == "__main__":
    shard_mlp_1d = SharedMLP1D(in_features=3)
    shard_mlp_2d = SharedMLP2D(in_features=3)

    # [batch=64, point_num=31, feature_num=3]
    points = torch.ones(64, 31, 3, dtype=shard_mlp_1d.dtype)
    points[:-1:2] = -1
    points[-1] = 0

    # [64, 3, 31]
    result_1d = shard_mlp_1d(points.permute(0, 2, 1))
    # [64, 1, 31, 3]
    result_2d = shard_mlp_2d(points.unsqueeze(dim=1))

    print(result_1d.shape)
    print(result_2d.shape)

    changed_1d = result_1d.squeeze().permute(0, 2, 1)
    changed_2d = result_2d.squeeze()

    print((changed_1d == changed_2d).all())
