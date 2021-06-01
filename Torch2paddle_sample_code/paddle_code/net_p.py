import paddle
import paddle.nn as nn
from utils_p import adaptive_instance_normalization

class ResnetBlock(nn.Layer):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], 
                                                  mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)), 
                                        nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], 
                                                  mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvBlock(nn.Layer):
    def __init__(self, dim1, dim2):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], 
                                                  mode='reflect'),
                                        nn.Conv2D(dim1, dim2, (3, 3)),
                                        nn.ReLU())

    def forward(self, x):
        out = self.conv_block(x)
        return out


class DecoderNet(nn.Layer):
    def __init__(self):
        super(DecoderNet, self).__init__()

        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)

        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)

        self.convblock_11 = ConvBlock(64, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], 
                                                  mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))

    def forward(self, cF, sF):

        out = adaptive_instance_normalization(cF['r41'], sF['r41'])
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r31'], sF['r31'])
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r21'], sF['r21'])
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out


vgg = nn.Sequential(
    nn.Conv2D(3, 3, (1, 1)),
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.Pad2D([1, 1, 1, 1], mode='reflect'),
    nn.Conv2D(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Encoder(nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg_net = vgg
        vgg_net.set_dict(paddle.load('/workspace/visCVPR2021/ZBK/pre_trained/vgg_normalised.pdparams'))
        self.enc_1 = nn.Sequential(*list(
            vgg_net.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(
            vgg_net.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(
            vgg_net.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(
            vgg_net.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(
            vgg_net.children())[31:44])  # relu4_1 -> relu5_1

    def forward(self, x):
        out = {}
        x = self.enc_1(x)
        out['r11'] = x
        x = self.enc_2(x)
        out['r21'] = x
        x = self.enc_3(x)
        out['r31'] = x
        x = self.enc_4(x)
        out['r41'] = x
        x = self.enc_5(x)
        out['r51'] = x
        return out
