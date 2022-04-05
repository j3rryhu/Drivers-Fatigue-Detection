from torch.utils import data
from PIL import Image
from resnet import ResNet
from torch import optim
import torch as t
from torchvision.transforms import ToPILImage
from torchvision import transforms
from loss_function import LossWithEuler
from euler_angle_calculator import EulerAngleCalc
import os
import torch
import numpy as np
show = ToPILImage()
transform = transforms.Compose([
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
        for line in f.readlines():
            line = line.split()
            if len(line)<202:
                break
            fname = line[0]
            img_path = self.img_path+fname
            lm = list(map(float, line[1:197]))
            for i in range(0, len(lm), 2):
                lm[i] /= 75
                lm[i+1] /= 100  # avoid gradient explosion
            ea = euler_angle_calculation.calculate(lm)
            boundingbox = list(map(float, line[197: 201]))
            attribute = list(map(int, line[201: 207]))
            sample = [img_path, lm, boundingbox, attribute, ea]
            self.samples.append(sample)
        f.close()

    def __getitem__(self, index):
        img_path = self.samples[index][0]
        img = Image.open(img_path)
        img = transform(img)
        lm = self.samples[index][1]     # only extract the upper, lower eyelid, corner and upper, lower lips
        train_pts = [60, 62, 64, 66, 68, 70, 72, 74, 88, 90, 92, 94]
        training_lm = []
        for track_pt in train_pts:
            training_lm.append(lm[2*track_pt])
            training_lm.append(lm[2 * track_pt+1])
        training_lm = torch.tensor(training_lm)
        boundingbox = self.samples[index][2]
        boundingbox = torch.tensor(boundingbox)
        attribute = self.samples[index][3]
        attribute = torch.tensor(attribute)
        ea = self.samples[index][4]
        ea = np.squeeze(ea)
        ea = torch.tensor(ea)
        return img, training_lm, boundingbox, attribute, ea

    def __len__(self):
        return len(self.samples)


trainloader = data.DataLoader(TrainFaces(transform), batch_size=32, shuffle=True, num_workers=0)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
net = ResNet().to(device)
criterion = LossWithEuler(attr_num)
optimizer = optim.SGD(net.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=8e-6)
t.set_num_threads(8)
# train on previous model
ckpt = torch.load('./models/ckpt.pth.tar')
state = ckpt['state']
model = ResNet()
model.load_state_dict(state)    # use for second round of training

if t.cuda.is_available():
    pass
else:
    exit()

correct = 0.0
total = 0.0
count = 0
print('all preparation done, start training')
for epoch in range(250):
    state = {
        "epoch": epoch+1,
        "state": net.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    print('-----epoch {}-----'.format(epoch))
    for i, data in enumerate(trainloader, 0):
        count += 1
        inputs, landmark, bbox, attr, euler_angle = data
        batch_size = inputs.size(0)
        landmark = t.tensor(landmark)
        landmark = landmark.to(device)
        attr = t.tensor(attr)
        attr = attr.to(device)
        euler_angle = t.tensor(euler_angle)
        euler_angle = euler_angle.to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, landmark, euler_angle, attr, batch_size)
        assert torch.isnan(loss).sum()==0, print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print("The {} batch loss is {}".format(i, loss.item()))
    t.save(state, '.\\models\\ckpt.pth.tar')

