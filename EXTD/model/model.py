import torch
import torch.nn as nn
import torch.nn.functional as F

from EXTD.base import BaseModel


class ConvolutionLayer(nn.Module):
    """ConvolutionLayer class"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int):
        """Instantiating ConvolutionLayer class

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.
            groups (int): Number of blocked connections from input channels to output channels.
        """
        super(ConvolutionLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        x = self.conv2d(x)

        return x


class FE_Block(nn.Module):
    """FE_Block class"""
    def __init__(self, stride: int, h: int):
        """Instantiating FE_Block class

        Args:
            stride (int): Stride of the convolution.
            h (int): Input channel width.
        """
        super(FE_Block, self).__init__()
        self.conv2d = ConvolutionLayer(3, h, 3, stride, 1, 1)
        self.norm = nn.BatchNorm2d(h)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(self.norm(x))

        return x


class Init_IRB_Block(nn.Module):
    """Init_IRB_Block class"""
    def __init__(self, stride: int, h: int, c: int):
        """Instantiating Init_IRB_Block class

        Args:
            stride (int): Stride of the convolution.
            h (int): Input channel width.
            c (int): Output channel width.
        """
        super(Init_IRB_Block, self).__init__()
        self.conv2d_1 = ConvolutionLayer(h, h, 3, stride, 1, 1)
        self.conv2d_2 = ConvolutionLayer(h, c, 1, stride, 0, 1)
        self.norm_1 = nn.BatchNorm2d(h)
        self.norm_2 = nn.BatchNorm2d(c)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.leaky_relu(self.norm_1(x))
        x = self.conv2d_2(x)
        x = self.norm_2(x)

        return x


class IRB_Block(nn.Module):
    """IRB_Block class"""

    def __init__(self, stride: int, h: int, c: int):
        """Instantiating IRB_Block class

        Args:
            stride (int): Stride of the convolution.
            h (int): Input channel width.
            c (int): Output channel width.
        """
        super(IRB_Block, self).__init__()
        self.conv2d_1 = ConvolutionLayer(c, h, 1, 1, 0, 1)
        self.conv2d_2 = ConvolutionLayer(h, h, 3, stride, 1, h)
        self.conv2d_3 = ConvolutionLayer(h, c, 1, 1, 0, 1)
        self.norm_1 = nn.BatchNorm2d(h)
        self.norm_2 = nn.BatchNorm2d(c)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.leaky_relu(self.norm_1(x))
        x = self.conv2d_2(x)
        x = F.leaky_relu(self.norm_1(x))
        x = self.conv2d_3(x)
        x = self.norm_2(x)

        return x


class Upsampling_Block(nn.Module):
    """Upsampling_Block class"""
    def __init__(self, scale_factor: int, stride: int, h: int, c: int):
        """Instantiating Upsampling_Block class

        Args:
            scale_factor (int): the multiplier for the image height / width
            stride (int): Stride of the convolution.
            h (int): Input channel width.
            c (int): Output channel width.
        """
        super(Upsampling_Block, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        self.conv2d_1 = ConvolutionLayer(c, h, 3, stride, 1, 1)
        self.conv2d_2 = ConvolutionLayer(h, c, 1, stride, 0, 1)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = F.leaky_relu(self.norm(x))

        return x


class Maxout(nn.Module):
    """Maxout class"""
    def __init__(self, d_in, d_out, pool_size):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class Classification_Block(nn.Module):
    """Classification_Block class"""
    def __init__(self, stride: int, c: int, maxout: int):
        """Instantiating Classification_Block class

        Args:
            stride (int): Stride of the convolution.
            c (int): Output channel width.
            maxout (int):
        """
        super(Classification_Block, self).__init__()
        self.maxout_bool = maxout
        self.maxout = Maxout(4, 2, 1)
        self.conv2d_1 = ConvolutionLayer(c, 4, 3, stride, 1, 1)
        self.conv2d_2 = ConvolutionLayer(c, 2, 3, stride, 1, 1)

    def forward(self, x):
        if self.maxout == 1:
            x = self.conv2d_1(x)
            x = self.maxout(x)
        else:
            x = self.conv2d_2(x)

        return x


class Regression_Block(nn.Module):
    """Regression_Block class"""
    def __init__(self, stride: int, c: int):
        """Instantiating Regression_Block class

        Args:
            stride (int): Stride of the convolution.
            c (int): Output channel width.
        """
        super(Regression_Block, self).__init__()
        self.conv2d = ConvolutionLayer(c, 4, 3, stride, 1, 1)

    def forward(self, x):
        x = self.conv2d(x)

        return x


class EXTDModel(nn.Module):
    """EXTDModel Layer"""
    def __init__(self):
        super(EXTDModel, self).__init__()
        self.h = 64
        self.c = 64
        self.fe_block = FE_Block(2, self.h)
        self.init_irb_block = Init_IRB_Block(1, self.h, self.c)
        self.irb_block_1 = IRB_Block(1, self.h * 2, self.c)
        self.irb_block_2 = IRB_Block(1, self.h * 4, self.c)
        self.irb_block_3 = IRB_Block(2, self.h * 4, self.c)
        self.upsampling_block = Upsampling_Block(2, 1, self.h, self.c)
        self.maxout_classification_block = Classification_Block(1, self.c, 1)
        self.classification_block = Classification_Block(1, self.c, 0)
        self.regression_block = Regression_Block(1, self.c)

    def forward(self, x):
        f = []
        cls = []
        reg = []
        previous_irb = self.fe_block(x)

        for i in range(6):
            init_irb = self.init_irb_block(previous_irb)
            irb = self.irb_block_1(init_irb)
            irb = self.irb_block_1(irb)
            irb = self.irb_block_1(irb)
            irb = self.irb_block_2(irb)
            irb = self.irb_block_3(irb)

            previous_irb = irb
            f.append(previous_irb)

        g = self.upsampling(f)
        for i in g:
            if i is g[-1]:
                cls.append(self.maxout_classification_block(i))
            else:
                cls.append(self.classification_block(i))
            reg.append(self.regression_block(i))

        pred_cls = torch.cat(cls, 0)
        pred_regs = torch.cat(reg, 0)
        return pred_cls, pred_regs

    def upsampling(self, f):
        g = []
        N = len(f)
        g.append(f[N-1])
        for i in range(N-1):
            g.append(self.upsampling_block(g[i]) + f[N-2-i])

        return g

e = EXTDModel()
e(torch.randn(1, 3, 640, 640))

pytorch_total_params = sum(p.numel() for p in e.parameters())
print(pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in e.parameters() if p.requires_grad)
print(pytorch_total_params)
