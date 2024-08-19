"""
coding:utf-8
@Time       :2024/8/15 11:10
@Author     :ywLi
@Institute  :DonghaiLab
"""
import torch

from torch import nn

# ----------------------------------------#
# Patch嵌入
# ----------------------------------------#
class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_chans=3, num_features=768, norm_layer=None,
                 flatten=True):
        super(PatchEmbed, self).__init__()
        # 14 * 14
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        # batch, 3, 224, 224 -> batch, 768, 14, 14
        x = self.proj(x)

        # batch, 768, 14, 14 -> batch, 768, 196 -> batch, 196, 768
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, input_shape=[224, 224], patch_size=16, in_chans=3, num_classes=1000, num_features=768,
            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU
    ):
        super(VisionTransformer, self).__init__()
        # -----------------------------------------------#
        #   batch, 224, 224, 3 -> batch, 196, 768
        # -----------------------------------------------#
        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                      num_features=num_features)
        num_patches = (224 // patch_size) * (224 // patch_size)
        self.num_features = num_features
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(224 // patch_size), int(224 // patch_size)]

        # --------------------------------------------------------------------------------------------------------------------#
        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        # --------------------------------------------------------------------------------------------------------------------#
        #   1, 1, 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))
        # --------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        # --------------------------------------------------------------------------------------------------------------------#
        #   1, 197, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # -----------------------------------------------#
        #   12次 batch, 197, 768 -> batch, 197, 768
        # -----------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )
        # 第一个norm写在了PatchEmbed里
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # batch, 3, 224, 224 -> batch, 196, 768
        x = self.patch_embed(x)

        # 1, 1, 768 -> batch, 1, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # batch, 1, 768 + batch, 196, 768 -> batch, 197, 768
        x = torch.cat((cls_token, x), dim=1)

        # 1, 1, 768
        cls_token_pe = self.pos_embed[:, 0:1, :]

        # 1, 196, 768
        img_token_pe = self.pos_embed[:, 1:, :]

        # 1, 196, 768 -> 1, 14, 14, 768 -> 1, 768, 14, 14
        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)

        # 1, 768, 14, 14 -> 以防输入图片大小不是224*224 做插值 -> 1, 768, 14, 14
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)

        # 1， 768， 14， 14 -> 1, 14, 14, 768 -> 1, 196, 768
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)

        # 1, 1, 768 + 1, 196, 768 -> 1, 197, 768
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        # batch, 197, 768 + 1, 197, 768 -> batch, 197, 768
        x = self.pos_drop(x + pos_embed)

        # 12次 batch, 197, 768
        x = self.blocks(x)

        # batch, 197, 768
        x = self.norm(x)

        # batch, 768
        return x[:, 0]