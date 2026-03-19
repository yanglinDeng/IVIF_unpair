import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, inter_dim, drop_rate):
        super().__init__()
        self.in_features = num_input_features
        self.drop_rate = drop_rate

        self.change = nn.Sequential(
            nn.BatchNorm2d(self.in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_features , inter_dim*2,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inter_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim*2, inter_dim,kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.change(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, inter_dim, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.dense_block = nn.ModuleList(
            [DenseLayer(num_input_features+i*inter_dim, inter_dim, drop_rate) for i in range(num_layers)])
    def forward(self,x):
        for layer in self.dense_block:
            x = layer(x)
        return x

class down_sample(nn.Module):
    def __init__(self,in_dim):
        super(down_sample, self).__init__()
        self.downsampler = nn.Sequential(
            nn.Conv2d(int(in_dim), int(in_dim//2), 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
    def forward(self,x):
        return self.downsampler(x)

class up_sample(nn.Module):
    def __init__(self,in_dim):
        super(up_sample, self).__init__()
        self.upsampler = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(in_dim//4), int(in_dim//2), 3, 1, 1)
        )
    def forward(self,x):
        return self.upsampler(x)

class eca_layer_1d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops



class FusionBlock(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU,drop = 0., use_eca=True):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear1(x)
        x = self.eca(x)
        x = self.linear2(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = H, w = W)
        return x


class Generator_net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = self.in_dim
        self.inchange = nn.Sequential(
            nn.Conv2d(self.dim,self.dim*8,1,1),
            nn.GroupNorm(2,self.dim*8),
            nn.LeakyReLU()
        )
        self.fus_0 = FusionBlock(self.dim * 8, self.dim * 32)

        self.dense_1 = DenseBlock(2, self.dim * 8, self.dim * 2, 0.)
        self.down_1 = down_sample(self.dim * 12)
        self.fus_1 = FusionBlock(self.dim * 24, self.dim * 96)

        self.dense_2 = DenseBlock(2, self.dim * 24, self.dim * 2, 0.)
        self.down_2 = down_sample(self.dim * 28)
        self.fus_2 = FusionBlock(self.dim * 56, self.dim * 224)

        self.dense_3 = DenseBlock(2, self.dim * 56, self.dim * 4, 0.)
        self.up_1 = up_sample(self.dim * 64)

        self.dense_4 = DenseBlock(2, self.dim * 56, self.dim * 4, 0.)
        self.up_2 = up_sample(self.dim * 64)

        self.outchange = nn.Sequential(
            nn.Conv2d(self.dim * 41, self.dim * 16, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim * 16, self.out_dim, 1, 1),
            nn.Tanh()
        )

    def forward(self,x):
        stage_0 = self.inchange(x)
        fuse_0 = self.fus_0(stage_0)

        stage_1 = self.down_1(self.dense_1(stage_0))
        fuse_1 = self.fus_1(stage_1)

        stage_2 = self.down_2(self.dense_2(stage_1))
        fuse_2 = self.fus_2(stage_2)

        stage_3 = self.up_1(self.dense_3(stage_2+fuse_2))
        stage_4 = self.up_2(self.dense_4(torch.cat((stage_3,stage_1+fuse_1),dim=1)))
        stage_5 = self.outchange(torch.cat((stage_4,stage_0+fuse_0,x),dim=1))

        return stage_5
