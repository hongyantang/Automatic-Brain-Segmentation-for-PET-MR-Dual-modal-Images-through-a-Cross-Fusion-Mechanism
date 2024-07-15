
from typing import Tuple

import torch.nn as nn
import torch

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXNet_3D.uxnet_encoder import uxnet_conv
from attention import CA, conv

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


# class ResBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  in_planes: int,
#                  planes: int,
#                  spatial_dims: int = 3,
#                  stride: int = 1,
#                  downsample: Union[nn.Module, partial, None] = None,
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#
#         super().__init__()
#
#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#
#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         residual = x
#
#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class UXNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.attention1 = CA(head_dim = feat_size[0])   #48
        self.attention2 = CA(head_dim = feat_size[1])   #96
        self.attention3 = CA(head_dim = feat_size[2])   #192
        self.attention4 = CA(head_dim = feat_size[3])   #384
        self.attention5 = CA(head_dim = feat_size[3]*2) #768
        self.conv1 = nn.Sequential(*[conv(feat_size[0] * 4, feat_size[0] * 2, 3)])   #96
        self.conv2 = nn.Sequential(*[conv(feat_size[1] * 4, feat_size[1] * 2, 3)])   #192
        self.conv3 = nn.Sequential(*[conv(feat_size[2] * 4, feat_size[2] * 2, 3)])   #384
        self.conv4 = nn.Sequential(*[conv(feat_size[3] * 4, feat_size[3] * 2, 3)])   #768     
        self.conv96 = nn.Conv3d(feat_size[0] * 2, feat_size[0] * 2, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True)
        self.conv192 = nn.Conv3d(feat_size[1] * 2, feat_size[1] * 2, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True)
        self.conv384 = nn.Conv3d(feat_size[2] * 2, feat_size[2] * 2, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True)
        self.conv768 = nn.Conv3d(feat_size[3] * 2, feat_size[3] * 2, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        '''
        x_1 = x_in[:,0:1,:,:,:]
        x_2 = x_in[:,1:2,:,:,:]
        outs_1 = self.uxnet_3d(x_1)
        outs_2 = self.uxnet_3d(x_2)

        enc1 = self.encoder1(x_1)
        enc2= self.encoder2(outs_1[0])
        enc3 = self.encoder3(outs_1[1])
        enc4 = self.encoder4(outs_1[2])

        enc_hidden_1 = self.encoder5(outs_1[3])
        enc_hidden_2 = self.encoder5(outs_2[3])
        enc5_1 = self.attention5(enc_hidden_1, enc_hidden_2)
        enc5_2 = self.attention5(enc_hidden_2,enc_hidden_1)
        enc5= self.conv4(torch.cat((enc5_1, enc5_2), 1))
        enc5= enc5+self.conv768(enc5)        
        # print(x.shape)
        dec3 = self.decoder5(enc5, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        
        return self.out(out)
        '''
        x_1 = x_in[:,0:1,:,:,:]
        x_2 = x_in[:,1:2,:,:,:]
        outs_1 = self.uxnet_3d(x_1)
        outs_2 = self.uxnet_3d(x_2)
        enc1 = self.encoder1(x_in)
        #x2 = self.attention2(outs_1[0], outs_2[0])
        x2_1 = self.encoder2(outs_1[0])
        x2_2 = self.encoder2(outs_2[0])
        enc2_1 = self.attention2(x2_1, x2_2)
        enc2_2 = self.attention2(x2_2,x2_1)
        enc2= self.conv1(torch.cat((enc2_1, enc2_2), 1))
        enc2= enc2+self.conv96(enc2)
        #x3 = self.attention3(outs_1[1], outs_2[1])
        x3_1 = self.encoder3(outs_1[1])
        x3_2 = self.encoder3(outs_2[1])
        enc3_1 = self.attention3(x3_1, x3_2)
        enc3_2 = self.attention3(x3_2,x3_1)
        enc3= self.conv2(torch.cat((enc3_1, enc3_2), 1))
        enc3= enc3+self.conv192(enc3)
        #x4 = self.attention4(outs_1[2], outs_2[2])
        x4_1 = self.encoder4(outs_1[2])
        x4_2 = self.encoder4(outs_2[2])
        enc4_1 = self.attention4(x4_1, x4_2)
        enc4_2 = self.attention4(x4_2,x4_1)
        enc4= self.conv3(torch.cat((enc4_1, enc4_2), 1))
        enc4= enc4+self.conv384(enc4)
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        #out_f = self.attention5(outs_1[3], outs_2[3])
        enc_hidden_1 = self.encoder5(outs_1[3])
        enc_hidden_2 = self.encoder5(outs_2[3])
        enc5_1 = self.attention5(enc_hidden_1, enc_hidden_2)
        enc5_2 = self.attention5(enc_hidden_2,enc_hidden_1)
        enc5= self.conv4(torch.cat((enc5_1, enc5_2), 1))
        enc5= enc5+self.conv768(enc5)        
        # print(x.shape)
        dec3 = self.decoder5(enc5, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        
        # feat = self.conv_proj(dec4)
        
        return self.out(out)
