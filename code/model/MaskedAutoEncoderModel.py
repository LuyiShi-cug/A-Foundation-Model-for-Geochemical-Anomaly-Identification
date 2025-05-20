import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MaskedViTAutoEncoder(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads,
                 num_layers, mask_ratio=0.75):
        super(MaskedViTAutoEncoder, self).__init__()
        
        # 基本参数设置
        self.image_size = image_size      # 输入图像大小，例如9
        self.patch_size = patch_size      # patch大小，例如3
        self.in_channels = in_channels    # 输入通道数，例如39
        self.embed_dim = embed_dim        # 嵌入维度，例如128
        self.mask_ratio = mask_ratio      # 掩码比例，例如0.75
        
        # 计算patch数量：(9/3)^2 = 9个patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding: 将图像转换为patches并嵌入
        # 输入: [B, 39, 9, 9] -> 输出: [B, 128, 3, 3]
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position embedding: 为每个patch添加位置编码
        # 形状: [1, 9, 128] - 9个patches，每个128维
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,        # 模型维度，例如128
            nhead=num_heads,          # 注意力头数，例如8
            dim_feedforward=embed_dim * 2,  # FFN维度，例如256
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,    # Transformer层数，例如6
            norm=nn.LayerNorm(embed_dim)
        )
        
        decoder_dim = embed_dim // 2
        decoder_num_layers = num_layers // 2
        # Transformer解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_num_layers,
            norm=nn.LayerNorm(decoder_dim)
        )
        
        self.encoder_proj = nn.Linear(embed_dim, decoder_dim)

        # mask token: 用于替换被mask的patches
        # 形状: [1, 1, 128]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 重建头: 将特征转换回原始patch大小
        # 输入: [B, L, 128] -> 输出: [B, L, patch_size^2*in_channels]
        # 例如: [B, 9, 128] -> [B, 9, 9*39]
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # 初始化mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 应用其他权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x):
        """随机mask输入patches"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # 输入x: [B, 39, 9, 9]
        
        # Patch embedding
        # [B, 39, 9, 9] -> [B, 128, 3, 3]
        x = self.patch_embed(x)
        # [B, 128, 3, 3] -> [B, 9, 128]
        x = x.flatten(2).transpose(1, 2)
        
        # 随机mask
        # x_masked: [B, num_keep, 128], 例如num_keep=3
        # mask: [B, 9], 0表示保留，1表示mask
        # ids_restore: [B, 9], 用于恢复原始顺序
        x_masked, mask, ids_restore = self.random_masking(x)
        
        # 添加位置编码
        # [B, num_keep, 128] + [1, num_keep, 128]
        x_masked = x_masked + self.pos_embed[:, :x_masked.size(1)]
        
        # Transformer编码
        # [B, num_keep, 128] -> [B, num_keep, 128]
        latent = self.encoder(x_masked)
        
        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # x: [B, num_keep, 128], ids_restore: [B, 9]
        
        # 添加mask tokens
        # [B, 9-num_keep, 128]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # 拼接: [B, num_keep, 128] + [B, 9-num_keep, 128] -> [B, 9, 128]
        x_ = torch.cat([x, mask_tokens], dim=1)
        # 恢复原始顺序: [B, 9, 128]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # 添加位置编码
        # [B, 9, 128] + [1, 9, 128]
        x = x_ + self.pos_embed
        
        x = self.encoder_proj(x)
        # Transformer解码
        # [B, 9, 128] -> [B, 9, 128]
        x = self.decoder(x)
        
        # 预测原始patch
        # [B, 9, 128] -> [B, 9, patch_size^2*in_channels]
        x = self.decoder_pred(x)
        
        # 重塑为图像patches
        # [B, 9, 9*39] -> [B, 3, 3, 3, 3, 39]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.in_channels)
        # 调整维度顺序: [B, 3, 3, 3, 3, 39] -> [B, 39, 9, 9]
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(B, self.in_channels, H * self.patch_size, W * self.patch_size)
        
        return x

    def forward(self, x, mask_ratio=None):
        # 输入x: [B, 39, 9, 9]
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
        
        # 编码
        # latent: [B, num_keep, 128]
        # mask: [B, 9]
        # ids_restore: [B, 9]
        latent, mask, ids_restore = self.forward_encoder(x)
        
        # 解码
        # [B, num_keep, 128] -> [B, 39, 9, 9]
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, mask

    def get_latent(self, x):
        """获取输入数据的潜在表示"""
        x = self.patch_embed(x)
        # [B, 128, 3, 3] -> [B, 9, 128]
        x = x.flatten(2).transpose(1, 2)
        
        # 添加位置编码
        # [B, num_keep, 128] + [1, num_keep, 128]
        x = x + self.pos_embed[:, :x.size(1)]
        
        return self.encoder(x)


    def compute_loss(self, x, pred, mask):
        """
        计算MAE损失
        Args:
            x: 原始输入 [B, C, H, W]
            pred: 模型预测 [B, C, H, W]
            mask: 掩码 [B, num_patches]
        """
        # 将输入和预测转换为patches
        x_patches = self.patchify(x)
        pred_patches = self.patchify(pred)
        
        # 计算patch级别的MSE
        loss = (pred_patches - x_patches) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per patch
        
        # 只计算被mask的patches的损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def patchify(self, imgs):
        """将图像转换为patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_channels))
        return x

    def train_chanel_loss(self, x, mask_ratio=None):
        # 输入x: [B, 39, 9, 9]
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
        b,f,w,h = x.shape
        # 编码
        # latent: [B, num_keep, 128]
        # mask: [B, 9]
        # ids_restore: [B, 9]
        chanel_loss=[]
        for i in range(f): #39
            latent, mask, ids_restore = self.forward_encoder(x[:,i].unsqueeze(1))
            # 解码
            # [B, num_keep, 128] -> [B, 39, 9, 9]
            pred = self.forward_decoder(latent, ids_restore)
            loss=self.compute_loss(x[:,i].unsqueeze(1), pred, mask)
            chanel_loss.append(loss)
        loss = torch.mean(torch.stack(chanel_loss))

        return loss

    def calculate_latent(self, x, mask_ratio=None):
        # 输入x: [B, 39, 9, 9]
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio

        b,f,w,h = x.shape

        # 编码
        # latent: [B, num_keep, 128]
        # mask: [B, 9]
        # ids_restore: [B, 9]

        all_latent=[]
        for i in range(f): #39
            latent = self.get_latent(x[:,i].unsqueeze(1))

            # 解码
            # [B, num_keep, 128] -> [B, 39, 9, 9]
            all_latent.append(latent.unsqueeze(1))
        all_latent = torch.cat(all_latent, dim=1)
        return all_latent

# 辅助函数
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

