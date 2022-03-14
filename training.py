import numpy as np
from torch.utils import data
from PIL import Image
from resnet import ResNet
from torch import nn
from torch import optim
import torch as t
from torchvision.transforms import ToPILImage
from torchvision import transforms
from loss_function import LossWithEuler
from euler_angle_calculator import EulerAngleCalc
import os
import cv2
from data_augmentation import DataAug
show = ToPILImage()
transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ann = open('./Img Dataset/annotations/train_annotation.txt')
annotations = ann.readlines()
attr_num = list(map(int, annotations[-7:-1]))
euler_angle_calculation = EulerAngleCalc()


def createDir(parentdir, dirname):
    if dirname not in os.listdir(parentdir):
        os.mkdir(dirname)


createDir('./', 'models')


class TrainFaces(data.Dataset):
    def __init__(self, trans):
        self.img_path = './Img Dataset/imgs/train/'
        f = open('./Img Dataset/annotations/train_annotation.txt', 'r')
        self.samples = []
        self.trans = trans
        dataaug = DataAug(do_random_crop=True)
        for line in f.readlines():
            line = line.split()
            if len(line)<202:
                break
            fname = line[0]
            path = self.img_path+fname
            img = Image.open(path)
            img = self.trans(img)
            lm = list(map(float, line[1:197]))
            ea = euler_angle_calculation.calculate(lm)
            boundingbox = list(map(int, line[197: 201]))
            attribute = list(map(int, line[201: 207]))
            sample = [img, lm, boundingbox, attribute, ea]
            self.samples.append(sample)
        self.samples = dataaug.preprocess_images(self.samples)
        f.close()

    def __getitem__(self, index):
        img = self.samples[index][0]
        lm = self.samples[index][1]
        boundingbox = self.samples[index][2]
        attribute = self.samples[index][3]
        ea = self.samples[index][4]
        return img, lm, boundingbox, attribute, ea

    def __len__(self):
        return len(self.samples)


trainloader = data.DataLoader(TrainFaces(transfrom), batch_size=128, shuffle=True, num_workers=0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
net = ResNet().to(device)
criterion = LossWithEuler(attr_num)
optimizer = optim.SGD(net.parameters(), lr=0.0005)
t.set_num_threads(8)

if t.cuda.is_available():
    pass
else:
    exit()

correct = 0.0
total = 0.0
count = 0
for epoch in range(150):
    state = {
        "epoch": epoch+1,
        "state": net.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    for i, data in enumerate(trainloader, 0):
        count += 1
        inputs, landmark, bbox, attr, euler_angle = data
        landmark = t.tensor(landmark)
        landmark = landmark.cuda()
        attr = t.tensor(attr)
        attr = attr.cuda()
        euler_angle = t.tensor(euler_angle)
        euler_angle = euler_angle.cuda()
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, landmark, euler_angle, attr)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("The {} batch loss is {}".format(i, loss.item()))
    t.save(state, '.\\models\\ckpt.pth.tar')
