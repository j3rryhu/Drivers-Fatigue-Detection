import torch
import torch.nn as nn
import math


class LossWithEuler(nn.Module):
    def __init__(self, attributes_num):
        super(LossWithEuler, self).__init__()
        self.attribute_num = torch.tensor(attributes_num) # num of samples in different attr

    def forward(self, inp, label, ea, attribute, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = batch_size
        attr_weight_mat = torch.zeros([batch_size, 1])
        angle_weight_mat = torch.zeros([batch_size, 1])
        attr_weight_mat.to(device)
        angle_weight_mat.to(device)
        for i in range(0, batch_size):
            attr_weight = 0
            for j in range(0, 6):
                if attribute[i][j] == 1:
                    attr_weight += sum(self.attribute_num)/self.attribute_num[j]
                    attr_weight_mat[i][0] = attr_weight
        for i in range(0, batch_size):
            angle_weight = 0
            for j in range(0, 3):
                angle_weight += 1 - math.cos(ea[i][j])
                angle_weight_mat[i][0] = angle_weight
        weight = torch.mul(angle_weight_mat, attr_weight_mat)
        weight = weight.to(device)
        mseloss = torch.mean(weight*torch.pow(inp - label, 2))
        assert torch.isnan(inp).sum()==0, print(inp)
        return mseloss











