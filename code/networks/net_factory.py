from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3
from networks.VNet import VNet, MCNet3d_v1, MCNet3d_v2,VNet_2out,VNet_2out_2,VNet_3out,VNet_4out,VNet_5out
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.unet_3D import unet_3D

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "3D-Unet":
        net = unet_3D(in_channels=in_chns, n_classes=class_num).cuda()
    if net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(in_channels=in_chns, n_classes=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "test":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "train":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_2out_2" and mode == "test":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out_2" and mode == "train":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_3out" and mode == "test":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_3out" and mode == "train":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_4out" and mode == "test":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_4out" and mode == "train":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_5out" and mode == "test":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_5out" and mode == "train":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    return net
