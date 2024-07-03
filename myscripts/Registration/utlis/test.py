import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import SimpleITK as sitk
import numpy as np


class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = 1
        self.conv_32 = self.__make_layer(32)
#         self.conv_64 = self.__make_layer(64)
#         self.conv_128 = self.__make_layer(128)
#         self.conv_256 = self.__make_layer(256)
#         self.conv_256s = self.__make_layer(256)
#         self.fc1 = nn.Linear(256*16*16, 4096)
        self.fc2 = nn.Linear(256*256*32, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, channels):
        layers = [nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False), nn.ReLU()]
        self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_32(x)
        out = F.max_pool2d(out, 2)
#         out = self.conv_64(out)
#         out = F.max_pool2d(out, 2)
#         out = self.conv_128(out)
#         out = F.max_pool2d(out, 2)
#         out = self.conv_256(out)
#         out = F.max_pool2d(out, 2)
#         out = self.conv_256s(out)
#         out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = F.relu(out)
        out = self.fc2(out)
        
        return out


def loss_function(out_flip, out_img, out_flip_inv, img_torch):
    return F.l1_loss(out_flip, out_img) + F.l1_loss(out_flip_inv, img_torch)


def transform(img, theta):
    theta = theta.reshape(2, 3)
    grid = F.affine_grid(theta.unsqueeze(0), img.size())
    output = F.grid_sample(img, grid)
    return output


def train():
    epochs = 1000
    all_loss = []
    for i in range(epochs):
        theta = model(img_torch)
        tmp = torch.eye(3).cuda()
        tmp[:2, :] = theta.reshape(2, 3)
        theta_inv = torch.inverse(tmp)[:2, :]
        out_img = transform(img_torch, theta)
        out_flip = torch.flip(out_img, dims=[-1])
        out_flip_inv = transform(out_flip, theta_inv)
        l = loss_function(out_flip, out_img, out_flip_inv, img_torch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        all_loss.append(l.detach().cpu().numpy())
        print(f"[{i+1}/{epochs}\t loss: {l}")




if __name__ == '__main__':
    file_path = r"/home/wyh/Codes/dataset/PRoVe/ProVe-IT-01-002/mCTA1_brain.nii.gz"
    img = sitk.GetArrayFromImage(sitk.ReadImage(file_path, sitk.sitkFloat32))
    img_torch = torch.from_numpy(img[149]).unsqueeze(0).unsqueeze(0).cuda()
    lr = 1e-3
    model = VGG(num_classes=6).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train()
    print()
