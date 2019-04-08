import torch
import torch.nn as nn

from .bspline_transformation import BSplineTransformation


class DeepSEA(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            nn.Dropout(p=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.5))

        self._spline_df = 14
        self.spline_tr = BSplineTransformation(self._spline_df)

        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.spline_tr(out3)
        reshape_out = out4.view(out4.size(0), 960 * self._spline_df)
        predict = self.classifier(reshape_out)
        return predict

def criterion():
    """
    The criterion the model aims to minimize.
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
