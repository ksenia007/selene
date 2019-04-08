import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from .bspline import bs


class BSplineTransformation(nn.Module):

    def __init__(self, degrees_of_freedom):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            radius = int(spatial_dim / 2)
            if spatial_dim % 2 == 0:
                dist = np.array(
                    list(range(-1 * radius, 0)) +
                    list(range(1, radius + 1)))
                dist = (np.log(np.abs(dist)) *
                        np.array([-1] * radius + [1] * radius))
            else:
                dist = np.array(
                    list(range(-1 * radius, 0)) +
                    list(range(1, radius + 2)))
                dist = (np.log(np.abs(dist)) *
                        np.array([-1] * radius + [1] * (radius + 1)))
            knots = [-3, -2.5, -2, -1, -0.5, 0.5, 1, 2, 2.5, 3]
            self._spline_tr = torch.from_numpy(bs(
                dist, knots=knots, intercept=True)).float()
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()

        batch = input.data
        transformed_batch = []
        for b in batch:
            output = b.mm(self._spline_tr)
            transformed_batch.append(output)
        transformed_batch = torch.stack(transformed_batch, 0)
        return Variable(transformed_batch)


