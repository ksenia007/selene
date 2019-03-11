import torch
import torch.nn as nn


class Basenji(nn.Module):

    def __init__(self, n_targets):
        super(Basenji, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 312, kernel_size=23, padding=11),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05))
        self.conv2 = nn.Sequential(
            nn.Conv1d(312, 368, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05))
        self.conv3 = nn.Sequential(
            nn.Conv1d(368, 435, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.05))
        self.conv4 = nn.Sequential(
            nn.Conv1d(435, 514, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.05))
        self.conv5 = nn.Sequential(
            nn.Conv1d(514, 607, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.05))
        self.conv6 = nn.Sequential(
            nn.Conv1d(607, 717, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05))

        self.dconv1 = nn.Sequential(
            nn.Conv1d(717, 108, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))
        self.dconv2 = nn.Sequential(
            nn.Conv1d(825, 108, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))
        self.dconv3 = nn.Sequential(
            nn.Conv1d(933, 108, kernel_size=3, dilation=8, padding=8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))
        self.dconv4 = nn.Sequential(
            nn.Conv1d(1041, 108, kernel_size=3, dilation=16, padding=16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))
        self.dconv5 = nn.Sequential(
            nn.Conv1d(1149, 108, kernel_size=3, dilation=32, padding=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))
        self.dconv6 = nn.Sequential(
            nn.Conv1d(1257, 108, kernel_size=3, dilation=64, padding=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10))

        self.final_conv = nn.Sequential(
            nn.Conv1d(1365, 1365, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05))
        self.classifier = nn.Linear(1365, 2002)   # nn.Conv1d(1365, 2002, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        dconv_out1 = self.dconv1(out6)
        cat_out1 = torch.cat([out6, dconv_out1], 1)
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = torch.cat([cat_out1, dconv_out2], 1)
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = torch.cat([cat_out2, dconv_out3], 1)
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = torch.cat([cat_out3, dconv_out4], 1)
        dconv_out5 = self.dconv5(cat_out4)
        cat_out5 = torch.cat([cat_out4, dconv_out5], 1)
        dconv_out6 = self.dconv6(cat_out5)
        cat_out6 = torch.cat([cat_out5, dconv_out6], 1)
        final_conv_out = self.final_conv(cat_out6)

        current_size = final_conv_out.size()[-1]
        x = 65536 // current_size
        out = 1024 // x
        final_conv_out = final_conv_out[:, :, out:current_size-out-1]
        final_conv_out = final_conv_out.transpose(1, 2)

        predict = self.classifier(final_conv_out)
        predict = predict.contiguous().view(
            predict.size(0), 2002 * 495)
        predict = self.sigmoid(predict)
        return predict

def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.

    Returns
    -------
    torch.nn._Loss
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
