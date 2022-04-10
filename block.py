import torch
import torch.nn as nn


class SharedMLP1D(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        
        self.conv = nn.Conv1d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=1, bias=False
        )

        self.batch_norm = nn.BatchNorm1d(out_features, eps=0, momentum=0, affine=False)

        # set weight the same
        self.dtype = self.conv.weight.dtype
        weight = torch.ones(out_features, in_features)
        for i in range(1, out_features+1):
            weight[i-1] = i
        weight = weight.unsqueeze(dim=-1)
        self.conv.weight = nn.Parameter(weight.to(dtype=self.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [batch, channel] or [batch, in_channel, length]
        # for point cloud: [batch, feature_num] or [batch, in_feature_num, point_num]
        # output shape: [batch, out_channel, length]
        # for point cloud: [batch, out_feature_num, point_num]
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class SharedMLP2D(nn.Module):
    def __init__(
        self, in_features: int, out_features: int
    ) -> None:
        super().__init__()
        # weight shape: [out_channel, in_channel, ksize[0], ksize[1]]
        # for point-wise shared mlp, the shape is [out_features, 1, 1, in_features]
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_features,
            kernel_size=(1, in_features), stride=1, padding=0, bias=False
        )

        self.batch_norm = nn.BatchNorm2d(out_features, eps=0, momentum=0, affine=False)

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
        # x shape: [batch, out_feature, point_num, 1]
        x = self.batch_norm(x)
        return x


if __name__ == "__main__":
    shared_mlp_1d = SharedMLP1D(in_features=3, out_features=16)
    shared_mlp_2d = SharedMLP2D(in_features=3, out_features=16)

    # 64 examples, [batch=64, point_num=30, feature=3]
    points = torch.ones(64, 30, 3, dtype=shared_mlp_1d.dtype)

    result_1d: torch.Tensor = shared_mlp_1d(points.permute(0, 2, 1))
    result_2d: torch.Tensor = shared_mlp_2d(points.unsqueeze(dim=1))

    print(result_1d.shape)
    print(result_2d.shape)
    
    # change shape to [batch, point_num, feature_num]
    changed_1d = result_1d.permute(0, 2, 1)
    changed_2d = result_2d.squeeze().permute(0, 2, 1)

    print((changed_1d == changed_2d).all())
