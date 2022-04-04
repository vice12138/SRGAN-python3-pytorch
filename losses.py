//这个模块定义了损失函数

import torch
from torch import nn
from torchvision.models.vgg import vgg16

class mseloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,real,fake):
        return self.mse(real,fake)


class vggloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
        vgg = vgg16(pretrained=True)
        lossnet = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in lossnet.parameters():
            param.requires_grad = False
        self.lossnetwork = lossnet
    def forward(self,real,fake):
        return self.mse( self.lossnetwork(real), self.lossnetwork(fake) )


class adversarialloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,fake_out):
        return torch.mean(1-fake_out)
class adversariallosslog(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,fake_out):
        return torch.mean( -torch.log(fake_out) )



class tvloss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = 3*(h_x-1)*w_x
        count_w = 3*h_x*(w_x-1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size






class dloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,real_out,fake_out):
        return torch.mean(1-real_out+fake_out)

class dlosslog(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,real_out,fake_out):
        return -torch.mean( torch.log(real_out)+torch.log(1-fake_out) )





class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

