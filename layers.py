import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



train_size = (1,3,256,256)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp: 
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        pools, convs, dynas = [],[],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(DynamicLAM(k))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class DynamicLAM(nn.Module):
    def __init__(self, inchannels, kernel_size=3):
        super(DynamicLAM, self).__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.inchannels = inchannels

        # adaptive weights generation
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inchannels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels // 4, inchannels * kernel_size * kernel_size, kernel_size=1, bias=False)
        )
        self.act = nn.Tanh()

        # adaptive fusion
        self.alpha_beta_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inchannels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels // 4, 2, kernel_size=1, bias=False)
        )
        self.softmax = nn.Sigmoid()

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # initialize weight_generator Conv2d layers
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.alpha_beta_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        n, c, h, w = x.size()
        identity = x

        # dynamic_weights generation
        dynamic_weights = self.weight_generator(x)  # [n, c * k * k, 1, 1]
        dynamic_weights = dynamic_weights.view(n, c, self.kernel_size * self.kernel_size)
        dynamic_weights = self.act(dynamic_weights)

        #  unfold operation
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)  # [n, c * k * k, L]
        x_unfold = x_unfold.view(n, c, self.kernel_size * self.kernel_size, h * w)  # [n, c, k * k, h * w]

        # apply dynamic weights
        out = dynamic_weights.unsqueeze(-1) * x_unfold  # [n, c, k * k, h * w]
        out = out.sum(dim=2)  # [n, c, h * w]
        out = out.view(n, c, h, w)
        out = self.act(out)

        # adaptive fusion
        fusion_weights = self.alpha_beta_generator(x)  # [n, 2, 1, 1]
        fusion_weights = fusion_weights.view(n, 2)
        fusion_weights = self.softmax(fusion_weights)  # 确保 alpha + beta = 1
        alpha = fusion_weights[:, 0].view(n, 1, 1, 1)
        beta = fusion_weights[:, 1].view(n, 1, 1, 1)

        # fusion
        out = alpha * out + beta * identity

        return out

class GatingModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GatingModule, self).__init__()
        self.in_channels = in_channels
        gating_channels = in_channels
        self.reduction = reduction

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(320, gating_channels, kernel_size=1, bias=False)
        self.convx = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(gating_channels, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, in_channels,bias=False),
            nn.Sigmoid()
        )


    def forward(self, x, gating_features):
        B, C, _, _ = x.size()
        x = self.convx(x)
        gating_pooled = self.global_pool(self.conv(gating_features)).view(B, -1)  # [B, C_g]

        gating = self.mlp(gating_pooled)
        out = x * gating.view(B, C, 1, 1).expand_as(x)  # [B, C, H, W]

        return out



class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_path='ckpt.pth'):
        super(MobileNetFeatureExtractor, self).__init__()
        mobilenet_v2 = models.mobilenet_v2(pretrained=False)
        mobilenet_v2.classifier = nn.Identity()

        state = torch.load(pretrained_path)
        new_state_dict = {k.replace('module.', ''): v for k, v in state.items()}
        mobilenet_v2.load_state_dict(new_state_dict,strict=False)

        self.features = mobilenet_v2.features

        self.stage1 = self.features[0:2]   
        self.stage2 = self.features[2:4]   
        self.stage3 = self.features[4:7]   
        self.stage4 = self.features[7:11]  
        self.stage5 = self.features[11:14] 
        self.stage6 = self.features[14:17] 
        self.stage7 = self.features[17:18] 
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        return x  