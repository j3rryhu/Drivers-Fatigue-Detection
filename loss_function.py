import torch
import torch.nn as nn
import math


class LossWithEuler(nn.Module):
    def __init__(self, attributes_num):
        super(LossWithEuler, self).__init__()
        self.attribute_num = torch.tensor(attributes_num) # num of samples in different attr

    def forward(self, inp, label, ea, attribute):
        _, batch_size = torch.Size(inp)
        attr_weight_mat = torch.zeros([1, batch_size])
        angle_weight_mat = torch.zeros([1, batch_size])
        for i in range(0, batch_size):
            attr_weight = 0
            for j in range(0, 6):
                if attribute[j][i] == 1:
                    attr_weight += sum(self.attribute_num)/self.attribute_num[i]
                    attr_weight_mat[1][i] = attr_weight
        for i in range(0, batch_size):
            angle_weight = 0
            for j in range(0, 3):
                angle_weight += 1 - math.cos(ea[j][i])
                angle_weight_mat[1][i] = angle_weight
        weight = torch.mul(angle_weight_mat, attr_weight_mat)
        mseloss = torch.mean(weight*torch.pow(inp - label, 2))
        return mseloss





