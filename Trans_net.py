import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        x = self.net(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()

        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(2,dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.mlp_dim=64
        self.dim_head = 8
        self.attn = PreNorm(dim, ChannelAttention(dim,1,8 ))
        self.ff = PreNorm(dim, FeedForward(dim, self.mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer_Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        return x

class MaskedLinear(nn.Module):
    def __init__(self, length, k_size=3):
        super().__init__()
        self.length = length
        self.k_size = k_size
        self.linear = nn.Linear(length, length, bias=False)
        self.register_buffer('mask', self.create_mask(length, k_size))

    def create_mask(self, length, k_size):
        mask = torch.zeros(length, length)
        radius = (k_size - 1) // 2
        for i in range(length):
            start = max(0, i - radius)
            end = min(length, i + radius + 1)
            mask[i, start:end] = 1
        return mask

    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)


class eca_layer_1d(nn.Module):

    def __init__(self, channel, k_size=3, pool_kernel_size=3, pool_stride=1):
        super(eca_layer_1d, self).__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=(pool_kernel_size - 1) // 2)
        self.conv = MaskedLinear(channel, k_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(-1, -2)
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        x = x.transpose(-1, -2)


        return x * y.expand_as(x)


class FusionBlock(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0., use_eca=True, k_size=3):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, dim)
        )
        self.eca = eca_layer_1d(hidden_dim, k_size) if use_eca else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear1(x)
        x = self.eca(x)
        x = self.linear2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
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

class IR_Visible_Fusion_Model(nn.Module):
    def __init__(self,indim=2, outdim=1,num_heads=2,num_layers=1):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.proj = nn.Conv2d(self.indim, self.indim*8, 1, 1)
        self.fuse_0 = FusionBlock(self.indim * 8, self.indim * 32)


        self.trans_1 = Transformer_Encoder(self.indim*8, num_heads,num_layers)
        self.down_1 = down_sample(self.indim*8)
        self.fuse_1 = FusionBlock(self.indim*16,self.indim*48)

        self.trans_2 = Transformer_Encoder(self.indim*16, num_heads,num_layers)
        self.down_2 = down_sample(self.indim * 16)
        self.fuse_2 = FusionBlock(self.indim * 32, self.indim * 96)

        self.trans_3 = Transformer_Encoder(self.indim*64, num_heads,num_layers)
        self.up_1 = up_sample(self.indim * 64)

        self.trans_4 = Transformer_Encoder(self.indim*64, num_heads,num_layers)
        self.up_2 = up_sample(self.indim * 64)

        self.final = nn.Sequential(
            nn.Conv2d(self.indim*48,self.indim*16,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.indim * 16, self.outdim, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        stage_0 = self.proj(x)
        fus_1 = self.fuse_0(stage_0)
        stage_1 = self.down_1(self.trans_1(stage_0))
        fus_2 = self.fuse_1(stage_1)
        stage_2 = self.down_2(self.trans_2(stage_1))
        fus_3 = self.fuse_2(stage_2)
        stage_3 = self.up_1(self.trans_3(torch.cat((stage_2,fus_3),dim=1)))
        stage_4 = self.up_2(self.trans_4(torch.cat((stage_3,fus_2,stage_1),dim=1)))
        stage_5 = self.final(torch.cat((stage_4,stage_0,fus_1),dim=1))
        return stage_5
