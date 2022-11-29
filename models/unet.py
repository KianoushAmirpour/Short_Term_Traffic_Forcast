import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]  # height
    tensor_size = tensor.size()[2]
    diff = (tensor_size - target_size) // 2  # cropping from both size
    return tensor[:, :, diff: tensor_size - diff, diff: tensor_size - diff]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(105, 128)
        self.down_conv_2 = double_conv(128, 256)
        self.down_conv_3 = double_conv(256, 512)
        self.down_conv_4 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(1024, 512)
        self.up_conv_2 = double_conv(512, 256)
        self.up_conv_3 = double_conv(256, 128)

        self.out = nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1)

        # for m in self.modules():
        #     if isinstance(m,  nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        #         nn.init.constant_(m.bias, 0.0)

    def forward(self, x_0):
        # encoder part
        x1 = self.down_conv_1(x_0)    # skip 8,128,496,448
        x2 = self.max_pool_2x2(x1)    # 8,128,248,224
        x3 = self.down_conv_2(x2)     # skip 8,256,248,224
        x4 = self.max_pool_2x2(x3)    # 8,256,124,112
        x5 = self.down_conv_3(x4)     # skip 8,512,124,112
        x6 = self.max_pool_2x2(x5)    # 8,512,62,56
        x7 = self.down_conv_4(x6)     # skip 8,1024,62, 56

        # decoder part
        x = self.up_trans_1(x7)
        x = self.up_conv_1(torch.cat([x, x5], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x1], 1))
        x = self.out(x)

        return x
