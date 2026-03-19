import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x_ = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x_ = self.net(x_).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_

class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super().__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = num_chans * expan_att_chans
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 3, groups=embed_dim,padding=1)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 3, groups=embed_dim,padding=1)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.group_qkv(x).view(B, C, -1, H, W).transpose(1, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, -1, H * W)
        k = k.view(B, self.num_heads, -1, H * W)
        v = v.view(B, self.num_heads, -1, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, -1, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()
        x_ = self.group_fus(x_)
        return x_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim,128,128])
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.mlp_dim = 64
        self.dim_head = 8
        self.attn = PreNorm(dim, ChannelAttention(dim, 1, 8))
        self.ff = PreNorm(dim, FeedForward(dim, self.mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class Score_net(nn.Module):
    def __init__(self, image_size, num_channels, embed_dim, num_heads, num_layers, emb_dropout, dropout):
        super().__init__()

        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.in_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 32, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(num_channels * 32),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(num_channels * 32, num_channels, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.05, True),

        )
        self.proj = nn.Conv2d(num_channels, embed_dim, 3, 1,1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 24, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(embed_dim*24),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(embed_dim * 24, embed_dim, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()

        self.linear_layer = nn.Sequential(
            nn.LayerNorm(image_size[0] * image_size[1] * embed_dim),
            nn.Linear(image_size[0] * image_size[1] * embed_dim, 1),
            nn.Tanh()
        )
        self.out = nn.Sequential(
            nn.Conv2d(embed_dim,1,1,1)
        )

    def forward(self, x):
        x = self.proj(self.in_conv(x))
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.flatten(self.final_conv(x))
        sc = self.linear_layer(x)
        # sc = self.out(self.final_conv(x))
        return sc

class Discriminator_net(nn.Module):
    def __init__(self, image_size=(128, 128), num_channels=1, embed_dim=3, num_heads=4):
        super().__init__()
        self.dis = Score_net(image_size, num_channels=num_channels, embed_dim=embed_dim, num_heads=num_heads, num_layers=1, emb_dropout=0., dropout=0)

    def forward(self, x):
        return self.dis(x)