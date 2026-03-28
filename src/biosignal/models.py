"""
EEG classification model architectures from TorchEEG.

Extracted from torcheeg==1.1.3 to avoid GNN dependency chain (torch_scatter).
These are pure PyTorch nn.Module implementations.

Sources:
  - TSCeption: Ding et al., "TSCeption: Capturing Temporal Dynamics and
    Spatial Asymmetry from EEG for Emotion Recognition", 2021.
    https://arxiv.org/abs/2104.02935
  - EEGNet: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
    for EEG-Based Brain-Computer Interfaces", J. Neural Eng., 2018.
    https://arxiv.org/abs/1611.08024
"""

import torch
import torch.nn as nn


class TSCeption(nn.Module):
    """Multi-scale temporal-spatial convolution for EEG emotion recognition.

    Input: (batch, 1, num_electrodes, chunk_size)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_electrodes: int = 28,
        num_T: int = 15,
        num_S: int = 15,
        in_channels: int = 1,
        hid_channels: int = 32,
        num_classes: int = 2,
        sampling_rate: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8

        self.Tception1 = self._conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self._conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self._conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self._conv_block(
            num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self._conv_block(
            num_T, num_S,
            (int(num_electrodes * 0.5), 1),
            (int(num_electrodes * 0.5), 1),
            int(self.pool * 0.25),
            padding=(0, 0, 1, 0) if num_electrodes % 2 == 1 else 0)

        self.fusion_layer = self._conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hid_channels), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_classes))

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel, stride, pool_kernel, padding=0):
        return nn.Sequential(
            nn.ZeroPad2d(padding) if padding != 0 else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out


class _Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm weight constraint (used by EEGNet)."""

    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """Compact convolutional neural network for EEG classification.

    Input: (batch, 1, num_electrodes, chunk_size)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        chunk_size: int = 151,
        num_electrodes: int = 60,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        num_classes: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), stride=1,
                      padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            _Conv2dWithConstraint(
                F1, F1 * D, (num_electrodes, 1),
                max_norm=1, stride=1, padding=(0, 0),
                groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_2), stride=1,
                      padding=(0, kernel_2 // 2), bias=False,
                      groups=F1 * D),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0),
                      groups=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self._feature_dim(), num_classes, bias=False)

    def _feature_dim(self):
        with torch.no_grad():
            mock = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock = self.block1(mock)
            mock = self.block2(mock)
        return self.F2 * mock.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        return x
