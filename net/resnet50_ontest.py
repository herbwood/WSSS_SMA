import sys
sys.path.append("/home/junehyoung/code/wsss_dfa")
import torch 
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import numpy as np


def fc_init_weight(m):
    torch.nn.init.xavier_normal_(m.weight)
    torch.nn.init.zeros_(m.bias)

def zero_init_weight(m):
    torch.nn.init.zeros_(m.weight)
    torch.nn.init.zeros_(m.bias)


class MutualExclusiveAggregator(nn.Module):

    def __init__(self, feature_dim, m):
        super(MutualExclusiveAggregator, self).__init__()
        self.feature_dim = feature_dim
        self.m = m
        self.mea_layer = nn.Conv2d(self.feature_dim, self.m, kernel_size=1)
        # self.mea_layer.apply(zero_init_weight)

    def forward(self, x):
        
        b, c, h, w = x.shape 
        mea_out = self.mea_layer(x)
        mea_out = mea_out.reshape(b, self.m, -1)
        mea_out = nn.Softmax(2)(mea_out)

        x = x.reshape(b, c, -1)
        out = torch.einsum("ijk,ilk->ijl", [x, mea_out])
        out = torch.mean(out, dim=2)

        return out 


class Classifier(nn.Module):
    
    def __init__(self, class_num, bottleneck_dim):
        super(Classifier, self).__init__()
        self.fc = nn.utils.weight_norm(nn.Linear(bottleneck_dim, class_num), name='weight')
        # self.fc.apply(fc_init_weight)

    def forward(self, x):
        out = self.fc(x)
        
        return out 


class Net(nn.Module):

    def __init__(self, stride=16):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                        self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                        self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 512, 1, bias=False)
        self.side2 = nn.Conv2d(512, 512, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 512, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 512, 1, bias=False)
        # self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.mea_fg = MutualExclusiveAggregator(feature_dim=2048, m=256)
        self.mea_bg = MutualExclusiveAggregator(feature_dim=2048, m=256)
        self.cls_fg = Classifier(class_num=20, bottleneck_dim=2048)
        self.cls_bg = Classifier(class_num=20, bottleneck_dim=2048)
        self.cls_concat = nn.Linear(4096, 2048)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([
                                        #   self.classifier,
                                          self.mea_fg,
                                          self.mea_bg,
                                          self.cls_fg,
                                          self.cls_bg,
                                          self.cls_concat,
                                          self.side1,
                                          self.side2,
                                          self.side3,
                                          self.side4,
                                        ])

    def forward(self, x, cur_epoch, target_epoch):

        x1 = self.stage1(x)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        z_fg = self.mea_fg(x4) # (b, 2048)
        z_bg = self.mea_bg(hie_fea)
        
        fgfg = self.cls_fg(z_fg) # (b, 20)
        bgfg = self.cls_bg(z_fg)
        fgbg = self.cls_fg(z_bg)
        bgbg = self.cls_bg(z_bg)

        z_concat = torch.cat([z_fg, z_bg.detach()], dim=1)
        z_concat = self.cls_concat(z_concat)
        fgbg_concat = self.cls_fg(z_concat)

        if cur_epoch >= target_epoch:
            indices = np.random.permutation(x.size(0))
            z_bg_shuffle = z_bg[indices]
            z_fg_shuffle = z_fg[indices]
        
            z_concat_shuffle = torch.cat([z_fg, z_bg_shuffle.detach()], dim=1)
            z_concat_shuffle2 = torch.cat([z_fg_shuffle, z_bg.detach()], dim=1)
            z_concat_shuffle = self.cls_concat(z_concat_shuffle)
            z_concat_shuffle2 = self.cls_concat(z_concat_shuffle2)
            fgbg_concat_shuffle = self.cls_fg(z_concat_shuffle)
            fgbg_concat_shuffle2 = self.cls_fg(z_concat_shuffle2)
            lambda6 = 1
            lambda8 = 1
            # lambda6 = cur_epoch / args.cam_num_epoches
        
        else:
            fgbg_concat_shuffle = torch.zeros_like(fgbg_concat)
            fgbg_concat_shuffle2 = torch.zeros_like(fgbg_concat)
            lambda6 = 0
            lambda8 = 0
            indices = np.arange(x.size(0))

        return fgfg, bgfg, fgbg, bgbg, fgbg_concat, fgbg_concat_shuffle, lambda6, z_fg, z_bg, fgbg_concat_shuffle2, lambda8, indices

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, stride=16, return_logit=False):
        super(CAM, self).__init__(stride=stride)
        
        self.return_logit = return_logit

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        if self.return_logit:
            mea_x = self.mea_fg(x)
            logit = self.cls_fg(mea_x)
            return logit.view(-1, 20)

        # (1, 2048, 32, 32)
        bs, nc, h, w = x.shape

        mea_x = self.mea_fg(x)
        logit = self.cls_fg(mea_x)

        param_fg = list(self.cls_fg.parameters())
        param_bg = list(self.cls_bg.parameters())
        weight_fg = np.squeeze(param_fg[-1].data) # (20, 2048)
        weight_bg = np.squeeze(param_bg[-1].data) # (20, 2048)

        cam_fg0 = torch.stack([torch.matmul(weight_fg[idx], x[0].reshape((nc, h * w))).reshape(h, w) for idx in range(0, 20)], 0)
        cam_fg1 = torch.stack([torch.matmul(weight_fg[idx], x[1].reshape((nc, h * w))).reshape(h, w) for idx in range(0, 20)], 0)
        cam_bg0 = torch.stack([torch.matmul(weight_bg[idx], x[0].reshape((nc, h * w))).reshape(h, w) for idx in range(0, 20)], 0)
        cam_bg1 = torch.stack([torch.matmul(weight_bg[idx], x[1].reshape((nc, h * w))).reshape(h, w) for idx in range(0, 20)], 0)

        cam_fg0 = F.relu(cam_fg0)
        cam_fg1 = F.relu(cam_fg1)
        cam_bg0 = F.relu(cam_bg0)
        cam_bg1 = F.relu(cam_bg1)

        cam_fg = cam_fg0 + cam_fg1.flip(-1) # (20, 32, 32)
        cam_bg = cam_bg0 + cam_bg1.flip(-1) # (20, 32, 32)

        cam_mix  = torch.cat([torch.where(torch.argmax(torch.cat([cam_fg[idx].unsqueeze(0), cam_bg[idx].unsqueeze(0)], dim=0), dim=0) == 0, cam_fg[idx].unsqueeze(0), torch.zeros_like(cam_fg[idx].unsqueeze(0))) for idx in range(0, 20)], 0)
        return cam_fg
        # return cam_mix
        # return cam_bg
    
if __name__ == "__main__":
    x = torch.randn(16, 3, 321, 321)
    model = Net()
    fgfg, bgfg, fgbg, bgbg, fgbg_concat, fgbg_concat_shuffle, lambda6, z_fg, z_bg, fgbg_concat_shuffle2, lambda8, indices = model(x, cur_epoch=0, target_epoch=1)
    print(z_fg.shape, z_bg.shape)
    print(z_fg.unsqueeze(1).shape, z_bg.unsqueeze(1).shape)