# ShardMLP
Pytorch implementations of Shared MLP

There is a lot of implementations of Shared MLP. No matter which api (conv1d, conv2d, Linear, and etc.) is used, as long as the manipulation (multiply, summation, and etc.) keep the same as the definition, the result is correct..

The simple code here offers both `nn.Conv1D` and `nn.Conv2D` implementations.

Besides, after Shared MLP, Batch Normalization is often applied on the output examples, so the code also offers both `nn.BatchNorm1D` and `nn.BatchNorm2D` implementations of batch normalization.
