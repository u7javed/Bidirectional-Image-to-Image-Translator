import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, conv_in, conv_out):
        super(ResBlock, self).__init__()

        self.res_sequence = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(conv_in, conv_out, kernel_size=(3, 3), stride=1, padding=0),
            nn.InstanceNorm2d(conv_out),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(conv_in, conv_out, kernel_size=(3, 3), stride=1, padding=0),
            nn.InstanceNorm2d(conv_out)
        )

    def forward(self, input):
        x = self.res_sequence(input)
        #element-wise sum of output with input
        output = input + x
        return output

class Generator(nn.Module):
    def __init__(self, channels, width, height, feature_space=64, n_res=16):
        super(Generator, self).__init__()

        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, feature_space, kernel_size=(7, 7), stride=1, padding=0),
            nn.InstanceNorm2d(feature_space),
            nn.ReLU(inplace=True)
        )

        #downsizing
        downsize_layer = []
        conv_in = feature_space
        conv_out = feature_space*2
        for _ in range(2):
           downsize_layer.append(nn.Conv2d(conv_in, conv_out, kernel_size=(3, 3), stride=2, padding=1))
           downsize_layer.append(nn.InstanceNorm2d(conv_out))
           downsize_layer.append(nn.ReLU(inplace=True))

           conv_in = conv_out
           conv_out = conv_out*2

        self.downsize_sequence = nn.Sequential(*downsize_layer)

        #residual layer
        residual_layer = []
        for _ in range(n_res):
            residual_layer.append(ResBlock(conv_in, conv_in))

        self.resid_sequence = nn.Sequential(*residual_layer)

        #upsizing
        upsize_layer = []
        conv_out = conv_in // 2
        for _ in range(2):
            upsize_layer.append(nn.Upsample(scale_factor=2))
            upsize_layer.append(nn.Conv2d(conv_in, conv_out, kernel_size=(3, 3), stride=1, padding=1))
            upsize_layer.append(nn.InstanceNorm2d(conv_out))
            upsize_layer.append(nn.ReLU(inplace=True))

            conv_in = conv_out
            conv_out = conv_out // 2

        self.upsize_sequence = nn.Sequential(*upsize_layer)

        #output conv
        self.output = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(conv_in, channels, kernel_size=(7, 7), stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.init_conv(input)
        x = self.downsize_sequence(x)
        x = self.resid_sequence(x)
        x = self.upsize_sequence(x)
        output = self.output(x)
        return output

class DiscriminatorBlock(nn.Module):
    def __init__(self, conv_in, conv_out, normalize):
        super(DiscriminatorBlock, self).__init__()

        self.normalize = normalize

        self.conv = nn.Conv2d(conv_in, conv_out, kernel_size=(4, 4), stride=2, padding=1)
        self.inst_norm = nn.InstanceNorm2d(conv_out)
        self.lr = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        x = self.conv(input)
        if self.normalize:
            x = self.inst_norm(x)
        output = self.lr(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, channels, width, height, feature_space=64):
        super(Discriminator, self).__init__()

        dis_block_layer = []
        #init discriminator block feed through
        dis_block_layer.append(DiscriminatorBlock(channels, feature_space, False))

        conv_in = feature_space
        for _ in range(3):
            dis_block_layer.append(DiscriminatorBlock(conv_in, conv_in*2, True))
            conv_in = conv_in * 2
        self.block_sequence = nn.Sequential(*dis_block_layer)

        self.output = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(conv_in, 1, kernel_size=(4, 4), stride=1, padding=1)
        )

    def forward(self, input):
        x = self.block_sequence(input)
        output = self.output(x)
        return output