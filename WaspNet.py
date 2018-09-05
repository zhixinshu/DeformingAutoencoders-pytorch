import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(torch.mul(input,mask), torch.mul(target,mask))
        return self.loss


class MaskedABSLoss(nn.Module):
    def __init__(self):
        super(MaskedABSLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(torch.mul(input,mask), torch.mul(target,mask))
        return self.loss


# an encoder architecture
class waspEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output   

# an encoder architecture
class waspEncoderReLU(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoderReLU, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output  

# an encoder architecture
class waspEncoderNosigmoid(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoderNosigmoid, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output   

# an encoder architecture
class waspEncoderInject(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128, injdim = 10):
        super(waspEncoderInject, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.injdim = injdim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim+injdim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim+self.injdim)
        #print(output.size())
        return output 

# an encoder architecture
class waspEncoderInject2(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32):
        super(waspEncoderInject2, self).__init__()
        self.opt = opt
        self.ngpu = ngpu
        self.injdim = opt.injdim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.opt.injdim + self.opt.zdim_inj + self.opt.zdim_inj, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.opt.injdim + self.opt.zdim_inj + self.opt.zdim_inj)
        #print(output.size())

        return output 

# a mixer (linear layer)
class waspMixer(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



# a mixer (linear layer)
class waspMixerHardtanh(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixerHardtanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Hardtanh(-1,1)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a mixer (linear layer)
class waspMixerNosig(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixerNosig, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a mixer (linear layer)
class waspMixerReLU(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixerReLU, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.ReLU()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a mixer (linear layer)
class waspSlicer(nn.Module):
    def __init__(self, opt, ngpu=1, pstart = 0, pend=1):
        super(waspSlicer, self).__init__()
        self.ngpu = ngpu
        self.pstart = pstart
        self.pend = pend
    def forward(self, input):
        output = input[:,self.pstart:self.pend].contiguous()
        return output


# a decoder architecture
class waspDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderExtraFit(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderExtraFit, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# a decoder architecture
class waspDecoderSigm(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderSigm, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoder2(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# a decoder architecture
class waspDecoderTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderHardTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderHardTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub),
            #nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoder_B(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder_B, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh_B(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh_B, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# warp image according to the grid
class waspWarper(nn.Module):
    def __init__(self, opt):
        super(waspWarper, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize = opt.imgSize

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output


# convolutional joint net for corr and decorr

class waspConvJoint0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

# convolutional joint net for corr and decorr

class waspConvResblockJoint0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvResblockJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 


# convolutional joint net for corr and decorr

class waspConvJoint9(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint9, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        #output = self.mixer(output)
        return output 


class waspConvJointConditional0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJointConditional0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.opt = opt
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz+self.opt.injdim,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2, inputc):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output0 = self.main(input0).view(-1,self.nz)
        output1 = torch.cat(output0, inputc,1)
        output = self.mixer(output1)
        return output 

# convolutional joint net for corr and decorr

class waspFadeJoint0(nn.Module):
    def __init__(self, opt,  nz1 =128, nz2 = 128, nz = 256):
        super(waspFadeJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.nz1 = nz1
        self.nz2 = nz2
        self.main = nn.Sequential(
            nn.Linear(self.nz1 + self.nz2, self.nz),
            nn.ReLU()
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0)
        output = self.mixer(output)
        return output 


# convolutional joint net for corr and decorr

class waspConvJointLsq0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJointLsq0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1)
            #nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

# convolutional joint net for corr and decorr

class waspConvJoint(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

class waspConvJoint2(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint2, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

class waspConvJoint3(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint3, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 


class waspConvJoint4(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint4, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 
# convolutional joint net for corr and decorr

class waspConvJointTanh(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32):
        super(waspConvJointTanh, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Tanh(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Tanh(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Tanh(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 128, 4, 4, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0)
        return output   


# The encoders
class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# The encoders
class Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp

# The encoders
class Encoders_MixSliceIntrinsic(nn.Module):
    def __init__(self, opt):
        super(Encoders_MixSliceIntrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zSmixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = opt.sdim)
        self.zTmixer = waspSlicer(opt, ngpu=1, pstart = opt.sdim, pend = opt.sdim + opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z        = self.encoder(input)
        self.zImg     = self.zImixer(self.z)
        self.zShade   = self.zSmixer(self.zImg)
        self.zTexture = self.zTmixer(self.zImg)
        self.zWarp    = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp

# The encoders
class Encoders_SliceIntrinsic(nn.Module):
    def __init__(self, opt):
        super(Encoders_SliceIntrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = opt.idim)
        self.zSmixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = opt.sdim)
        self.zTmixer = waspSlicer(opt, ngpu=1, pstart = opt.sdim, pend = opt.sdim + opt.tdim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = opt.idim, pend = opt.idim + opt.wdim)

    def forward(self, input):
        self.z        = self.encoder(input)
        self.zImg     = self.zImixer(self.z)
        self.zShade   = self.zSmixer(self.zImg)
        self.zTexture = self.zTmixer(self.zImg)
        self.zWarp    = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp

# The encoders
class EncodersNosig(nn.Module):
    def __init__(self, opt):
        super(EncodersNosig, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixerNosig(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixerNosig(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


# The encoders
class EncodersSlicer(nn.Module):
    def __init__(self, opt):
        super(EncodersSlicer, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim, pend = self.opt.idim + opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# The encoders
class EncodersSlicerNosig(nn.Module):
    def __init__(self, opt):
        super(EncodersSlicerNosig, self).__init__()
        self.opt= opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderNosig(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim, pend = self.opt.idim + opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


# The encoders
class EncodersInject(nn.Module):
    def __init__(self, opt):
        super(EncodersInject, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim, injdim = opt.injdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.injdim+opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.injdim+opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zLabel = self.z[:,0:self.opt.injdim]
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel

# The encoders
class EncodersInject2(nn.Module):
    def __init__(self, opt):
        super(EncodersInject2, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject2(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf)
        self.zLmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.zdim_inj, pend = (self.opt.zdim_inj+self.opt.injdim))
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = (self.opt.zdim_inj+self.opt.injdim))
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.zdim_inj, pend = (self.opt.zdim_inj+self.opt.zdim_inj+self.opt.injdim))
    def forward(self, input):
        self.z      = self.encoder(input)
        self.zLabel = self.zLmixer(self.z)
        self.zImg   = self.zImixer(self.z)
        self.zWarp  = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel

# The decoders
#class Decoders(nn.Module):
#    def __init__(self, opt):
#        super(Decoders, self).__init__()
#        self.ngpu     = opt.ngpu
#        self.idim = opt.idim
#        self.wdim = opt.wdim
#        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub = 1)
#        self.decoderW = waspDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=-1, ub=1)
#        self.warper   = waspWarper(opt)
#    def forward(self, zI, zW):
#        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
#        self.warping = self.decoderW(zW.view(-1,self.wdim,1,1))
#        self.output  = self.warper(self.texture, self.warping)
#        return self.texture, self.warping, self.output

# The decoders that use residule warper
class DecodersResWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersResWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output

# The decoders that use residule warper
class DecodersResWarperInject2(nn.Module):
    def __init__(self, opt):
        super(DecodersResWarperInject2, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim_inj, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim_inj, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.opt.idim_inj,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.opt.wdim_inj,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output


class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss

class L0SignLoss(nn.Module):
    def __init__(self,opt):
        super(L0SignLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.sign(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))) + 
            torch.sum(torch.sign(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))))
        return self.loss     

class L0SignLossNC(nn.Module):
    def __init__(self,opt):
        super(L0SignLossNC, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.FloatTensor(1).fill_(weight)
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.sign(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))) + 
            torch.sum(torch.sign(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))))
        return self.loss  

class L0SignLossMO(nn.Module):
    def __init__(self,opt):
        super(L0SignLossMO, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.dx = torch.sign(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        self.dy = torch.sign(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        self.dx,_ = torch.max(self.dx, dim = 1)
        self.dy,_ = torch.max(self.dy, dim = 1)
        self.loss = w * (torch.sum(self.dx) + 
            torch.sum(self.dy))
        return self.loss    


class L0SignLossFlex(nn.Module):
    def __init__(self,opt):
        super(L0SignLossFlex, self).__init__()
        self.opt = opt
        self.flex = 0.001
        self.cut1 = nn.Hardtanh(0,1)
        self.cut2 = nn.Hardtanh(0,1)
        self.cut3 = nn.Hardtanh(0,1)
        self.cut4 = nn.Hardtanh(0,1)
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.diff_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.diff_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.diff_x2 = x[:, :, :, 1:] - x[:, :, :, :-1]
        self.diff_y2 = x[:, :, 1:, :] - x[:, :, :-1, :]
        self.abs_diff_x = torch.abs(self.diff_x)
        self.abs_diff_y = torch.abs(self.diff_y)
        self.abs_diff_x2 = torch.abs(self.diff_x2)
        self.abs_diff_y2 = torch.abs(self.diff_y2)
        self.dx = self.abs_diff_x - self.flex
        self.dy = self.abs_diff_y - self.flex
        self.dx2 = self.abs_diff_x2 - self.flex
        self.dy2 = self.abs_diff_y2 - self.flex
        self.dx = self.cut1(self.dx)
        self.dy = self.cut2(self.dy)
        self.dx2 = self.cut3(self.dx2)
        self.dy2 = self.cut4(self.dy2)
        self.loss = w * (torch.sum(torch.sign(self.dx)) + 
            torch.sum(torch.sign(self.dy)))
        return self.loss  

        
class SelfSmoothLoss2(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss2, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.sum(torch.mul(self.x_diff, self.x_diff)) + torch.sum(torch.mul(self.y_diff, self.y_diff))
        self.loss = w * self.loss
        return self.loss  

class SelfSmoothLoss2MO(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss2MO, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.dx = torch.mul(self.x_diff, self.x_diff)
        self.dy = torch.mul(self.y_diff, self.y_diff)
        self.dx,_ = torch.max(self.dx, dim=1)
        self.dy,_ = torch.max(self.dy, dim=1)
        self.loss = torch.sum(self.dx) + torch.sum(self.dy)
        self.loss = w * self.loss
        return self.loss  


class SelfSmoothLoss(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.norm(self.x_diff) + torch.norm(self.y_diff)
        self.loss = w * self.loss
        return self.loss        

class WeightMSELoss(nn.Module):
    def __init__(self,opt):
        super(WeightMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.opt = opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss


class WeightABSLoss(nn.Module):
    def __init__(self,opt):
        super(WeightABSLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.opt=opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss

class WeightBCELoss(nn.Module):
    def __init__(self,opt):
        super(WeightBCELoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.opt=opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss

class BiasReduceLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss

class BiasReduceABSLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceABSLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.L1Loss()
    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss

class WaspGridSpatialIntegral0(nn.Module):
    def __init__(self,opt):
        super(WaspGridSpatialIntegral0, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,self.w,self.w).fill_(0)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,self.w).fill_(0)
        self.filterx[:,:,-1,:] = 1
        self.filtery[:,:,:,-1] = 1
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=self.w-1)
        fully = F.conv2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=self.w-1)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid


class WaspGridSpatialIntegral(nn.Module):
    def __init__(self,opt):
        super(WaspGridSpatialIntegral, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid

# The decoders that use residule warper
#class DecodersResWarper(nn.Module):
#    def __init__(self, opt):
#        super(DecodersResWarper, self).__init__()
#        self.ngpu = opt.ngpu
#        self.idim = opt.idim
#        self.wdim = opt.wdim
#        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
#        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
#        self.warper   = waspWarper(opt)
#    def forward(self, zI, zW, basegrid):
#        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
#        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
#        self.resWarping = self.resWarping*2-1
#        self.warping = self.resWarping + basegrid
#        self.output  = self.warper(self.texture, self.warping)
#        return self.texture, self.resWarping, self.output

# The decoders that use residule warper
class DecodersSleepyWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersSleepyWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.ReLUx = nn.ReLU()
        self.ReLUy = nn.ReLU()
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid

        self.warping_grad_x = self.warping[:, 0, :, 1:] - self.warping[:, 0, :, :-1] 
        self.warping_grad_y = self.warping[:, 1, 1:, :] - self.warping[:, 1, :-1, :]
        self.warping_grad_x_pos = self.ReLUx(self.warping_grad_x) 
        self.warping_grad_y_pos = self.ReLUy(self.warping_grad_y) 

        self.warping[:, 0, :, 1:] = self.warping[:, 0, :, :-1] + self.warping_grad_x_pos
        self.warping[:, 1, 1:, :] = self.warping[:, 1, :-1, :] + self.warping_grad_y_pos

        self.resWarping_pos = self.warping - basegrid
        
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping_pos, self.output





# The decoders that use residule warper
class DecodersIntegralWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.1
        self.warping = self.integrator(self.diffentialWarping)-2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper0(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper0, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoderSigm(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


# The decoders that use residule warper
class DecodersIntegralWarper1(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper1, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping



# The decoders that use residule warper
class waspIntrinsicComposer(nn.Module):
    def __init__(self, opt):
        super(waspIntrinsicComposer, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc
    def forward(self, shading, texture):
        self.shading = shading.repeat(1,self.nc,1,1)
        self.img = torch.mul(self.shading, texture)
        return self.img

# The decoders that use residule warper
class waspIntrinsicComposer2(nn.Module):
    def __init__(self, opt):
        super(waspIntrinsicComposer2, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc
    def forward(self, shading, texture):
        self.img = torch.mul(shading, texture)
        return self.img


# The decoders that use residule warper
class DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping


# The decoders that use residule warper
class DecodersIntegralWarper2_Intrinsic2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_Intrinsic2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.1, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.01, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.intrinsicComposer = waspIntrinsicComposer2(opt)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarperInject2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarperInject2, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim_inj, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim_inj, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.opt.idim_inj,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.opt.wdim_inj,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping




# The decoders that use residule warper
class DecodersIntegralWarper2_B(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_B, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder_B(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh_B(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper3(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper3, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.01
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


class DecodersSlopeWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersSlopeWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.negaSlope = nn.ReLU()
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid

        self.warping_grad_x = self.warping[:, 0, :, :-1] - self.warping[:, 0, :, 1:] 
        self.warping_grad_y = self.warping[:, 1, :-1, :] - self.warping[:, 1, 1:, :]
        #print(self.warping_grad_x.size())
        #print(self.warping_grad_y.size())
        self.warping_grad = torch.cat((self.warping_grad_x.unsqueeze(1), self.warping_grad_y.permute(0,2,1).unsqueeze(1)),1).contiguous()
        self.slope = self.negaSlope(self.warping_grad)
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.slope

# a mixer (linear layer)
class waspMaskComposer(nn.Module):
    def __init__(self, opt):
        super(waspMaskComposer, self).__init__()
        self.nc = opt.nc
    def forward(self, foreground, base, mask):
        #print(mask.size())
        matte = mask.repeat(1,self.nc,1,1)
        output = foreground*matte + base*(1-matte)
        return output

# a mixer (linear layer)
class waspMasker(nn.Module):
    def __init__(self, opt):
        super(waspMasker, self).__init__()
        self.nc = opt.nc
    def forward(self, img, mask):
        #print(mask.size())
        matte = mask.repeat(1,self.nc,1,1)
        output = img*matte
        return output

# The encoders
class Encoders_slice_mask(nn.Module):
    def __init__(self, opt):
        super(Encoders_slice_mask, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend=opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = opt.idim, pend=opt.idim + opt.wdim)
        self.zMmixer = waspSlicer(opt, ngpu=1, pstart = opt.idim + opt.wdim, pend=opt.idim + opt.wdim + opt.mdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zMask = self.zMmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zMask


# The encoders
class Encoders_mask(nn.Module):
    def __init__(self, opt):
        super(Encoders_mask, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        self.zMmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.mdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zMask = self.zMmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zMask



# The decoders that use residule warper
class DecodersIntegralWarper2_mask(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_mask, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.mdim = opt.mdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderM = waspDecoder(opt, ngpu=self.ngpu, nz=opt.mdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
        self.composer = waspMaskComposer(opt)
        self.pooler = nn.AvgPool2d(13,8,padding=6)
        self.masker = waspMasker(opt)
    def forward(self, zI, zW, zM,  basegrid, basesynth):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.mask = self.decoderM(zM.view(-1,self.mdim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.synth  = self.warper(self.texture, self.warping)
        self.output = self.composer(self.synth, basesynth, self.mask)
        self.mask_pooled = self.pooler(self.mask)
        self.texture_masked = self.masker(self.texture, self.mask)
        return self.texture, self.resWarping, self.output, self.warping, self.synth, self.mask_pooled, self.mask, self.texture_masked


# waspAffineGrid: Generate an affine grid from affine transform parameters. 
class waspAffineGrid(nn.Module):
    def __init__(self, opt):
        super(waspAffineGrid, self).__init__()
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize

    def forward(self, af_pars, basegrid):
#        output_grid = F.affine_grid(af_pars, torch.Size((self.batchSize, 3, self.imgSize, self.imgSize))).permute(0,3,1,2)
#        return output_grid
#def getAffineGrid(af_pars, basegrid):
        nBatch, nc, iS, _ = basegrid.size()
        affine = af_pars.expand(iS, iS, nBatch, 6).permute(2,3,0,1).contiguous()
        afft_x = affine[:,0,:,:]*basegrid[:,0,:,:] + affine[:,1,:,:]*basegrid[:,1,:,:] + affine[:,2,:,:]
        afft_y = affine[:,3,:,:]*basegrid[:,0,:,:] + affine[:,4,:,:]*basegrid[:,1,:,:] + affine[:,5,:,:]
        afft_x = afft_x.unsqueeze(1)
        afft_y = afft_y.unsqueeze(1)
        output_grid = torch.cat((afft_x, afft_y), 1)
        return output_grid


# Decoder for affine transform. Just a linear layer. 
class waspDecoderAffineLinear(nn.Module):
    def __init__(self, opt, ngpu=1, nz=10, ndim=6):
        super(waspDecoderAffineLinear, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is z. Just apply a linear layer on top. 
            nn.Linear(nz, ndim)
        )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))    
        else:
            output = self.main(input)
#        print(output.view(-1,2,3).size())
#        return output.contiguous().view(-1,2,3)
        return output


# Encoder that also encodes the affine transform.
class EncodersAffineIntegralSlice(nn.Module):
    def __init__(self, opt):
        super(EncodersAffineIntegralSlice, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim, pend=self.opt.idim+self.opt.wdim)
        self.zAmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim+self.opt.wdim, pend=self.opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zAffT = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffT

# Encoder that also encodes the affine transform.
class EncodersAffineIntegral(nn.Module):
    def __init__(self, opt):
        super(EncodersAffineIntegral, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        self.zAmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.adim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zAffT = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffT

# The decoders that use residule warper
class DecodersIntegralWarper2_mask2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_mask2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.mdim = opt.mdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderM = waspDecoder(opt, ngpu=self.ngpu, nz=opt.mdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
        self.composer = waspMaskComposer(opt)
        self.masker = waspMasker(opt)
    def forward(self, zI, zW, zM,  basegrid, basesynth):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.mask = self.decoderM(zM.view(-1,self.mdim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.synth  = self.warper(self.texture, self.warping)
        self.output = self.composer(self.synth, basesynth, self.mask)
        self.texture_masked = self.masker(self.texture, self.mask)
        return self.texture, self.resWarping, self.output, self.warping, self.synth,  self.mask, self.texture_masked


class L2Reg(nn.Module):
    def __init__(self, opt):
        super(L2Reg, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, zero, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(x, zero)
        return self.loss

# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersAffineIntegralWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersAffineIntegralWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode affine params, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Transform the texture. 
        self.af_tex  = self.warper(self.texture, self.affine)
        # Decode and integrate the warping grid. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply the warping grid to the deformed texture. 
        self.output  = self.warper(self.af_tex, self.warping)
        # Apply the warping grid to the transformed grid to get the final deformation field.
        self.warp_af = self.warper(self.affine, self.warping)
        # Get the residual deformation field.
        self.resWarping = self.warp_af - self.affine #self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarper2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper1 = waspWarper(opt)
        self.warper2 = waspWarper(opt)
        self.warper3 = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper1(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper2(self.wp_tex, self.affine)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper3(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarperPreTrain(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarperPreTrain, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.trW  = opt.trW
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        if self.trW:
            self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
            self.warping = self.integrator(self.differentialWarping)-1.2
            self.warping = self.cutter(self.warping)
            # Apply face deformation to texture. 
            self.wp_tex  = self.warper(self.texture, self.warping)
        else:
            self.wp_tex  = self.texture
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Check if we compute the non-linear deformation. 
        if self.trW:
            # Apply affine transformation to face warping to get the final deformation field.
            self.warp_af = self.warper(self.warping, self.affine)
            # Get the residual deformation.
            self.resWarping = self.warping - basegrid
        else:
            self.warp_af = self.affine
            # Residual warping is None if self.trW is False. 
            self.resWarping = None
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarperFixAffine(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarperFixAffine, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.trW  = opt.trW
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        if self.trW:
            self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
            self.warping = self.integrator(self.differentialWarping)-1.2
            self.warping = self.cutter(self.warping)
            # Apply face deformation to texture. 
            self.wp_tex  = self.warper(self.texture, self.warping)
        else:
            self.wp_tex  = self.texture
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Check if we compute the non-linear deformation. 
        if self.trW:
            # Apply affine transformation to face warping to get the final deformation field.
            self.warp_af = self.warper(self.warping, self.affine)
            # Get the residual deformation.
            self.resWarping = self.warping - basegrid
        else:
            self.warp_af = self.affine
            # Residual warping is None if self.trW is False. 
            self.resWarping = None
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# The encoders
class Encoders_AE(nn.Module):
    def __init__(self, opt):
        super(Encoders_AE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        return self.z

# The encoders
class Decoders_AE(nn.Module):
    def __init__(self, opt):
        super(Decoders_AE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = waspDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output     = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output

# The encoders for FreeNet
class Encoders_FreeNet(nn.Module):
    def __init__(self, opt):
        super(Encoders_FreeNet, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        return self.z

# The decoders for FreeNet
class Decoders_FreeNet(nn.Module):
    def __init__(self, opt):
        super(Decoders_FreeNet, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        #self.decoder = waspDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoder = waspReactor(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output   = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output

# The decoders for FreeNet
class Decoders_FreeNet2(nn.Module):
    def __init__(self, opt):
        super(Decoders_FreeNet2, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        #self.decoder = waspDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoder = waspReactor2(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output   = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output



# a decoder architecture
class waspReactor(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspReactor, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf*4, 1, 8),
            nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf*2, 1, 16),
            nn.BatchNorm2d(ngf * 2),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf, 1, 32),
            nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf, 1, 64),
            nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc + 2, 3, 1, 1, bias=False),
            ReformBlock(opt, nc, 1, 64),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# Define a reformer block
class ReformBlock(nn.Module):
    def __init__(self, opt, dim_feat, dim_warp, w):
        super(ReformBlock, self).__init__()
        self.w = w
        self.thresh1 = 0.2/self.w
        self.thresh2 = 3.1/self.w
        self.opt = opt
        self.warper = waspReformer(self.opt, self.w)
        self.integrator = waspGridIntegrator(self.opt, self.w)
        self.burner = nn.Hardtanh(self.thresh1,self.thresh2)
        self.cutter = nn.Hardtanh(-1,1)
        self.dim_feat = dim_feat
        self.dim_warp = dim_warp*2
    def forward(self, x):
        self.feat = x[ : , 0 : self.dim_feat, : , : ].contiguous()
        self.warp = x[ : , -self.dim_warp : , : , : ].contiguous()
        self.warp_burn = self.burner(self.warp)
        self.warp_inte = self.integrator(self.warp_burn)-1.2
        self.warp_cut = self.cutter(self.warp_inte)
        self.warpfeat = self.warper(self.feat, self.warp_cut)
        return self.warpfeat

# warp image according to the grid
class waspReformer(nn.Module):
    def __init__(self, opt, w):
        super(waspReformer, self).__init__()
        self.opt = opt
        self.w = w
        self.batchSize = opt.batchSize
    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output

class waspGridIntegrator(nn.Module):
    def __init__(self, opt, w):
        super(waspGridIntegrator, self).__init__()
        self.opt =opt
        self.w = w
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid


class waspReactor2(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspReactor2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf*4, 1, 8),
            nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf*2, 1, 16),
            nn.BatchNorm2d(ngf * 2),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            #ReformBlock(opt, ngf, 1, 32),
            nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf + 2, 4, 2, 1, bias=False),
            ReformBlock(opt, ngf, 1, 64),
            nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc + 2, 3, 1, 1, bias=False),
            ReformBlock(opt, nc, 1, 64),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# an encoder architecture
class waspEncoderVAE(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoderVAE, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )
        self.muMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
        self.logvarMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        mu = self.muMixer(output)
        logvar = self.logvarMixer(output)
        #print(output.size())
        return mu, logvar

# an encoder architecture
class waspEncoderVAEReLU(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoderVAEReLU, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.ReLU()
        )
        self.muMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
        self.logvarMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        mu = self.muMixer(output)
        logvar = self.logvarMixer(output)
        #print(output.size())
        return mu, logvar


# an encoder architecture
class waspEncoderVAENoSig(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoderVAENoSig, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            #nn.Sigmoid()
        )
        self.muMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
        self.logvarMixer = waspMixerNosig(opt, ngpu=1, nin=self.ndim, nout=self.ndim)
    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        mu = self.muMixer(output)
        logvar = self.logvarMixer(output)
        #print(output.size())
        return mu, logvar

# The encoders of VAE
class Encoders_VAE0(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE0, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = waspEncoderVAEReLU(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
    def forward(self, input):
        self.mu, self.logvar    = self.encoder(input)
        return self.mu, self.logvar


# The encoders of VAE
class Encoders_VAE(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = waspEncoderVAE(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
    def forward(self, input):
        self.mu, self.logvar    = self.encoder(input)
        return self.mu, self.logvar

# The encoders of VAE NoSig
class Encoders_VAEns(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAEns, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = waspEncoderVAENoSig(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
    def forward(self, input):
        self.mu, self.logvar    = self.encoder(input)
        return self.mu, self.logvar

class Reparameterizers_VAE(nn.Module):
    def __init__(self, opt):
        super(Reparameterizers_VAE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    def forward(self, input_mu, input_logvar):
        z = self.reparameterize(input_mu, input_logvar)
        return z

# The encoders of VAE
class Decoders_VAE(nn.Module):
    def __init__(self, opt):
        super(Decoders_VAE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = waspDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output     = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output


class KLDLoss(nn.Module):
    def __init__(self,opt):
        super(KLDLoss, self).__init__()
        self.opt=opt
    def forward(self, mu, logvar, weight=1):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.batchSize = mu.size(0)
        weight /= self.batchSize 
        self.loss = weight*KLD
        return self.loss


# The encoders
class Encoders_VAE2(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE2, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)

        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        
        self.muMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)
        self.logvarMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)

        self.muMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)
        self.logvarMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)

        self.zI_mu = self.muMixer_zI(self.zImg)
        self.zI_logvar = self.logvarMixer_zI(self.zImg)

        self.zW_mu = self.muMixer_zW(self.zWarp)
        self.zW_logvar = self.logvarMixer_zW(self.zWarp)

        return self.zI_mu, self.zI_logvar, self.zW_mu, self.zW_logvar

# The encoders
class Encoders_VAE3(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE3, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderReLU(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)

        self.zImixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        
        self.muMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)
        self.logvarMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)

        self.muMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)
        self.logvarMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)

        self.zI_mu = self.muMixer_zI(self.zImg)
        self.zI_logvar = self.logvarMixer_zI(self.zImg)

        self.zW_mu = self.muMixer_zW(self.zWarp)
        self.zW_logvar = self.logvarMixer_zW(self.zWarp)

        return self.zI_mu, self.zI_logvar, self.zW_mu, self.zW_logvar

class Reparameterizers_VAE2(nn.Module):
    def __init__(self, opt):
        super(Reparameterizers_VAE2, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.reparI = Reparameterizers_VAE(opt)
        self.reparW = Reparameterizers_VAE(opt)
    def forward(self, mu_I, logvar_I, mu_W, logvar_W):
        zI = self.reparI(mu_I, logvar_I)
        zW = self.reparW(mu_W, logvar_W)
        return zI, zW


# convolutional joint net for corr and decorr

class waspConv0(nn.Module):
    def __init__(self, opt,  nc=3, ndf = 32, nz = 256):
        super(waspConv0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input0):
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

class waspConv1_NoMixNoSig(nn.Module):
    def __init__(self, opt,  nc=3, ndf = 32, nz = 256):
        super(waspConv1_NoMixNoSig, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 4, 0, bias=False)
        )
        #self.mixer = nn.Sequential(
        #    nn.Linear(self.nz,1),
        #    nn.Sigmoid()
        #)
    def forward(self, input0):
        output = self.main(input0).view(-1,1)
        #output = self.mixer(output)
        return output 


class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
    def forward(self, output, label):
        self.loss = 0.5 * torch.mean((output-label)**2)
        return self.loss



# The encoders
class Encoders_VAE2I(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE2I, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderReLU(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)

        self.zImixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        
        self.muMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)
        self.logvarMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zW = self.zWmixer(self.z)

        self.zI_mu = self.muMixer_zI(self.zImg)
        self.zI_logvar = self.logvarMixer_zI(self.zImg)

        return self.zI_mu, self.zI_logvar, self.zW

# The encoders
class Encoders_VAE2W(nn.Module):
    def __init__(self, opt):
        super(Encoders_VAE2W, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderReLU(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)

        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
  
        self.muMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)
        self.logvarMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zI  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)

        self.zW_mu = self.muMixer_zW(self.zWarp)
        self.zW_logvar = self.logvarMixer_zW(self.zWarp)

        return self.zI, self.zW_mu, self.zW_logvar


# The code sampler
class SplitSampler(nn.Module):
    def __init__(self, opt):
        super(SplitSampler, self).__init__()
        self.ngpu = opt.ngpu
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.zI = self.zImixer(input)
        self.zW = self.zWmixer(input)
        return self.zI, self.zW

# The decoders that use residule warper
class Intrinsicers(nn.Module):
    def __init__(self, opt):
        super(Intrinsicers, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.tdim)
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
    def forward(self, z):
        self.zS = self.zSmixer(z)
        self.zT = self.zTmixer(z)
        self.shading = self.decoderS(self.zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(self.zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture) 
        return self.zS, self.zT, self.shading, self.texture, self.img

# The decoders that use residule warper
class IntrinsicersExtraFit(nn.Module):
    def __init__(self, opt):
        super(IntrinsicersExtraFit, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.tdim)
        self.decoderS = waspDecoderExtraFit(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.decoderT = waspDecoderExtraFit(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
    def forward(self, z):
        self.zS = self.zSmixer(z)
        self.zT = self.zTmixer(z)
        self.shading = self.decoderS(self.zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(self.zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture) 
        return self.zS, self.zT, self.shading, self.texture, self.img

# The decoders that use residule warper
class IntrinsicersColor(nn.Module):
    def __init__(self, opt):
        super(IntrinsicersColor, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.idim, nout = opt.tdim)
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.intrinsicComposer = waspIntrinsicComposer2(opt)
    def forward(self, z):
        self.zS = self.zSmixer(z)
        self.zT = self.zTmixer(z)
        self.shading = self.decoderS(self.zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(self.zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture) 
        return self.zS, self.zT, self.shading, self.texture, self.img


# The decoders that use residule warper
class Intrinsicers2(nn.Module):
    def __init__(self, opt):
        super(Intrinsicers2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.decoderS = waspDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.decoderT = waspDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf/2, lb=0.001, ub=1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
    def forward(self, z):
        self.zS = self.zSmixer(z)
        self.zT = self.zTmixer(z)
        self.shading = self.decoderS(self.zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(self.zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture) 
        return self.zS, self.zT, self.shading, self.texture, self.img

# an encoder architecture
class waspEncoderInject4(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32):
        super(waspEncoderInject4, self).__init__()
        self.opt = opt
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.opt.idim_inj + self.opt.wdim_inj + self.opt.injdim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.opt.idim_inj + self.opt.wdim_inj + self.opt.injdim)
        #print(output.size())
        return output 

class EncodersInject3(nn.Module):
    def __init__(self, opt):
        super(EncodersInject3, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject4(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf)
        self.zLmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim_inj, pend = (self.opt.idim_inj+self.opt.injdim))
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim_inj, pend = (self.opt.idim_inj+self.opt.wdim))
    def forward(self, input):
        self.z      = self.encoder(input)
        self.zLabel = self.zLmixer(self.z)
        self.zImg   = self.zImixer(self.z)
        self.zWarp  = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel

# a mixer (linear layer)
class waspSlicer2(nn.Module):
    def __init__(self, opt, ngpu=1, pstart1 = 0, pend1 = 1, pstart2 = 2, pend2 = 3):
        super(waspSlicer2, self).__init__()
        self.ngpu = ngpu
        self.pstart1 = pstart1
        self.pend1 = pend1
        self.pstart2 = pstart2
        self.pend2 = pend2
    def forward(self, input):
        output1 = input[:,self.pstart1:self.pend1]
        output2 = input[:,self.pstart2:self.pend2]
        output = torch.cat((output1,output2),dim=1).contiguous()
        return output

class EncodersInject4(nn.Module):
    def __init__(self, opt):
        super(EncodersInject4, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject4(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf)
        self.zLmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim_inj, pend = (self.opt.idim_inj+self.opt.injdim))
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim_inj, pend = (self.opt.idim_inj+self.opt.wdim))
        self.zSmixer = waspSlicer2(opt, ngpu=1, pstart1 = 0, pend1 = self.opt.idim_inj, pstart2 = self.opt.idim, pend2 = (self.opt.idim_inj+self.opt.wdim))
    def forward(self, input):
        self.z      = self.encoder(input)
        self.zLabel = self.zLmixer(self.z)
        self.zImg   = self.zImixer(self.z)
        self.zWarp  = self.zWmixer(self.z)
        self.zStyle = self.zSmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel, self.zStyle

# The decoders that use integral warper
class DecodersIntegralWarperInject3(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarperInject3, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.opt.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.opt.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping



########################################################
########################################################
################## Dense Net Variants ##################
########################################################
########################################################

# Dense block in encoder. 
# Dense block in encoder. 
class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no 
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

# Dense block in encoder. 
class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no 
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)
             

# an encoder architecture
# Densely connected convolutions. 
class waspDenseEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid, f_args=[]):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.BatchNorm2d(nc),
                nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 32 x 32
                DenseBlockEncoder(ndf, 6),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args),

                # state size. (ndf*2) x 16 x 16
                DenseBlockEncoder(ndf*2, 12),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args),

                # state size. (ndf*4) x 8 x 8
                DenseBlockEncoder(ndf*4, 24),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 4 x 4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output   

class waspDenseDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[0,1]):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*4),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf*4, 24),
            DenseTransitionBlockDecoder(ngf*4, ngf*2),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf*2, 12),
            DenseTransitionBlockDecoder(ngf*2, ngf),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseEncodersAffineIntegral(nn.Module):
    def __init__(self, opt):
        super(DenseEncodersAffineIntegral, self).__init__()
        self.opt        = opt
        self.ngpu       = opt.ngpu
        self.encoder    = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer    = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zWmixer    = waspSlicer(opt, ngpu=1, pstart=self.opt.idim, pend=self.opt.idim+self.opt.wdim)
        self.zAmixer    = waspSlicer(opt, ngpu=1, pstart=self.opt.idim+self.opt.wdim, pend=self.opt.zdim)
    def forward(self, inputs):
        self.z          = self.encoder(inputs)
        self.zImg       = self.zImixer(self.z)
        self.zWarp      = self.zWmixer(self.z)
        self.zAffine    = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffine

class DenseEncodersAffineIntegral_NoSlice(nn.Module):
    def __init__(self, opt):
        super(DenseEncodersAffineIntegral_NoSlice, self).__init__()
        self.opt        = opt
        self.ngpu       = opt.ngpu
        self.encoder    = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer    = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer    = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        self.zAmixer    = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.adim)
    def forward(self, inputs):
        self.z          = self.encoder(inputs)
        self.zImg       = self.zImixer(self.z)
        self.zWarp      = self.zWmixer(self.z)
        self.zAffine    = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffine


class DenseDecodersIntegralAffineWarper(nn.Module):
    def __init__(self, opt):
        super(DenseDecodersIntegralAffineWarper, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.adim       = opt.adim
        self.decoderI   = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.decoderA   = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper     = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# The encoders
class Dense_Encoders(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# The encoders
class Dense_Encoders_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zShade, self.zTexture, self.zWarp


class Dense_Encoders_Intrinsic_Baseline(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Intrinsic_Baseline, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zSmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.sdim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zShade = self.zSmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        return self.z, self.zShade, self.zTexture

# The decoders that use residule warper
class Dense_DecodersIntegralWarper2_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zS, zT, zW, basegrid):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.shading, self.texture, self.img, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class Dense_DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.img = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.img, self.resWarping, self.output, self.warping


# The decoders that use residule warper
class Dense_DecodersIntegralWarper2_Intrinsic_Baseline(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Intrinsic_Baseline, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.decoderS = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
    def forward(self, zS, zT):
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        return self.shading, self.texture, self.img


class DenseEncodersAffineIntegral_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(DenseEncodersAffineIntegral_Intrinsic, self).__init__()
        self.opt        = opt
        self.ngpu       = opt.ngpu
        self.encoder    = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer    = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zTmixer    = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.tdim)
        self.zSmixer    = waspSlicer(opt, ngpu=1, pstart=self.opt.tdim, pend=self.opt.tdim+self.opt.sdim)
        self.zWmixer    = waspSlicer(opt, ngpu=1, pstart=self.opt.idim, pend=self.opt.idim+self.opt.wdim)
        self.zAmixer    = waspSlicer(opt, ngpu=1, pstart=self.opt.idim+self.opt.wdim, pend=self.opt.zdim)
    def forward(self, inputs):
        self.z          = self.encoder(inputs)
        self.zImg       = self.zImixer(self.z)
        self.zTexture   = self.zTmixer(self.zImg)
        self.zShade     = self.zSmixer(self.zImg)
        self.zWarp      = self.zWmixer(self.z)
        self.zAffine    = self.zAmixer(self.z)
        return self.z, self.zImg, self.zShade, self.zTexture, self.zWarp, self.zAffine



class DenseDecodersIntegralAffineWarper_Intrinsic(nn.Module):
    def __init__(self, opt):
        super(DenseDecodersIntegralAffineWarper_Intrinsic, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.adim       = opt.adim
        self.sdim       = opt.sdim
        self.tdim       = opt.tdim
        self.decoderS   = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=1, ngf=opt.ngf, lb=0, ub=1, f_args=[0.01,1])
        self.decoderT   = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1, f_args=[0.01,1])
        self.decoderW   = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.decoderA   = waspDecoderAffineLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper     = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
    def forward(self, zS, zT, zW, zA, basegrid):
        # Decode the texture. 
        #self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.albedo  = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.texture = self.intrinsicComposer(self.shading, self.albedo)
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.shading, self.albedo, self.texture, self.resWarping, self.output, self.warp_af, self.af_pars





# The encoders
class Dense_Encoders_VAE3(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_VAE3, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim, f_activation=nn.ReLU)

        self.zImixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixerReLU(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)
        
        self.muMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)
        self.logvarMixer_zI = waspMixerNosig(opt, ngpu=1, nin=opt.idim, nout=opt.idim)

        self.muMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)
        self.logvarMixer_zW = waspMixerNosig(opt, ngpu=1, nin=opt.wdim, nout=opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)

        self.zI_mu = self.muMixer_zI(self.zImg)
        self.zI_logvar = self.logvarMixer_zI(self.zImg)

        self.zW_mu = self.muMixer_zW(self.zWarp)
        self.zW_logvar = self.logvarMixer_zW(self.zWarp)

        return self.zI_mu, self.zI_logvar, self.zW_mu, self.zW_logvar


# The encoders
class Dense_Encoders_VAE0(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_VAE0, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim, f_activation=nn.ReLU)
        
        self.muMixer_z = waspMixerNosig(opt, ngpu=1, nin=opt.zdim, nout=opt.zdim)
        self.logvarMixer_z = waspMixerNosig(opt, ngpu=1, nin=opt.zdim, nout=opt.zdim)


    def forward(self, input):
        self.z     = self.encoder(input)

        self.z_mu = self.muMixer_z(self.z)
        self.z_logvar = self.logvarMixer_z(self.z)

        return self.z_mu, self.z_logvar


# The encoders of VAE
class Dense_Decoders_VAE(nn.Module):
    def __init__(self, opt):
        super(Dense_Decoders_VAE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output     = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output


# The encoders
class Dense_Encoders_AE(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_AE, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        return self.z


# The encoders of VAE
class Dense_Decoders_AE(nn.Module):
    def __init__(self, opt):
        super(Dense_Decoders_AE, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.zdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
    def forward(self, input):
        self.output     = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output


#############################################
######### For lighting and normals  #########
#############################################

# The encoders
class Dense_Encoders_Light(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Light, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zLmixer = waspMixerNosig(opt, ngpu=1, nin = opt.zdim, nout = opt.ldim)
        self.zNmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.ndim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zLight = self.zLmixer(self.z)
        self.zNormals = self.zNmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zLight, self.zNormals, self.zTexture, self.zWarp


# a mixer (linear layer)
class waspAngleToDirection(nn.Module):
    def __init__(self, opt):
        super(waspAngleToDirection, self).__init__()
        self.opt = opt
    def forward(self, input):
        phi = input[:,0,:,:].unsqueeze(1)*3.1415926
        theta = input[:,1,:,:].unsqueeze(1)*3.1415926
        nx = torch.mul(torch.sin(phi), torch.cos(theta))
        ny = torch.cos(phi)
        nz = torch.mul(torch.sin(phi), torch.sin(theta))
        output = torch.cat((nx,ny,nz),dim=1).contiguous()
        return output



class getNormalImage(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(getNormalImage, self).__init__()
        self.opt = opt
    def forward(self, x):
        y  = (x+1)/2
        return y

class getNormalImage2(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(getNormalImage2, self).__init__()
        self.opt = opt
    def forward(self, x):
        y  = x
        y[:,0,:,:] = (y[:,0,:,:]+1)/2
        y[:,1,:,:] = (y[:,1,:,:]+1)/2
        return y       

class HomogeneousCoord(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(HomogeneousCoord, self).__init__()
        self.opt = opt
    def forward(self, x):
        y = Variable(torch.cuda.FloatTensor(x.size(0),1,x.size(2),x.size(3)).fill_(1).cuda(),requires_grad=False)
        z = torch.cat((x,y),1)
        return z

class MMatrixOld(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(MMatrixOld, self).__init__()
        self.opt = opt
    def forward(self, L):
        # input L:[batchSize,9]
        # output M: [batchSize, 4, 4]
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743152
        c4 = 0.886227
        c5 = 0.247708
        M00 = c1*L[:,8].unsqueeze(0)
        M01 = c1*L[:,4].unsqueeze(0)
        M02 = c1*L[:,7].unsqueeze(0)
        M03 = c2*L[:,3].unsqueeze(0)
        M10 = c1*L[:,4].unsqueeze(0)
        M11 = -c1*L[:,8].unsqueeze(0)
        M12 = c1*L[:,5].unsqueeze(0)
        M13 = c2*L[:,1].unsqueeze(0)
        M20 = c1*L[:,7].unsqueeze(0)
        M21 = c1*L[:,5].unsqueeze(0)
        M22 = c3*L[:,6].unsqueeze(0)
        M23 = c2*L[:,2].unsqueeze(0)
        M30 = c2*L[:,3].unsqueeze(0)
        M31 = c2*L[:,1].unsqueeze(0)
        M32 = c2*L[:,2].unsqueeze(0)
        M33 = c4*L[:,0].unsqueeze(0) - c5*L[:,6].unsqueeze(0)
        M0 = torch.cat((M00,M01,M02,M03),dim=1).unsqueeze(1)
        M1 = torch.cat((M10,M11,M12,M13),dim=1).unsqueeze(1)
        M2 = torch.cat((M20,M21,M22,M23),dim=1).unsqueeze(1)
        M3 = torch.cat((M30,M31,M32,M33),dim=1).unsqueeze(1)
        M = torch.cat((M0,M1,M2,M3),dim=1)
        return M

class MMatrix(nn.Module):
    """docstring for getHomogeneousCoord"""
    def __init__(self, opt):
        super(MMatrix, self).__init__()
        self.opt = opt
    def forward(self, L):
        # input L:[batchSize,9]
        # output M: [batchSize, 4, 4]
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743152
        c4 = 0.886227
        c5 = 0.247708
        M00 = c1*L[:,8].unsqueeze(1)
        M01 = c1*L[:,4].unsqueeze(1)
        M02 = c1*L[:,7].unsqueeze(1)
        M03 = c2*L[:,3].unsqueeze(1)
        M10 = c1*L[:,4].unsqueeze(1)
        M11 = -c1*L[:,8].unsqueeze(1)
        M12 = c1*L[:,5].unsqueeze(1)
        M13 = c2*L[:,1].unsqueeze(1)
        M20 = c1*L[:,7].unsqueeze(1)
        M21 = c1*L[:,5].unsqueeze(1)
        M22 = c3*L[:,6].unsqueeze(1)
        M23 = c2*L[:,2].unsqueeze(1)
        M30 = c2*L[:,3].unsqueeze(1)
        M31 = c2*L[:,1].unsqueeze(1)
        M32 = c2*L[:,2].unsqueeze(1)
        M33 = c4*L[:,0].unsqueeze(1) - c5*L[:,6].unsqueeze(1)
        M0 = torch.cat((M00,M01,M02,M03),dim=1).unsqueeze(1)
        M1 = torch.cat((M10,M11,M12,M13),dim=1).unsqueeze(1)
        M2 = torch.cat((M20,M21,M22,M23),dim=1).unsqueeze(1)
        M3 = torch.cat((M30,M31,M32,M33),dim=1).unsqueeze(1)
        M = torch.cat((M0,M1,M2,M3),dim=1)
        return M

# a mixer (linear layer)
class waspShadeRenderer(nn.Module):
    def __init__(self, opt):
        super(waspShadeRenderer, self).__init__()
        self.opt = opt
        self.getHomo = HomogeneousCoord(opt)
        self.getMMatrix = MMatrix(opt)
    def forward(self, light, normals):
        # homogeneous coordinate of the normals
        batchSize = normals.size(0)
        W = normals.size(2)
        H = normals.size(3)
        hNormals = self.getHomo(normals)
        # matrix for light
        mLight = self.getMMatrix(light)
        # get shading from these two: N x 4 , N = batchSize x W x H 
        hN_vec = hNormals.view(batchSize, 4, -1).permute(0,2,1).contiguous().view(-1,4)
        # N x 1 x 4
        hN_vec_Left  = hN_vec.unsqueeze(1)
        # N x 4 x 1
        hN_vec_Right = hN_vec.unsqueeze(2)
        # expand the lighting from batchSize x 4 x 4 to N x 4 x 4
        hL = mLight.view(batchSize,16).repeat(1,W*H).view(-1,4,4)
        shade0 = torch.matmul(hN_vec_Left, hL)
        shade1 = torch.matmul(shade0, hN_vec_Right)
        #shade1 is tensor of size Nx1x1 = batchSize x W x H
        shading = shade1.view(batchSize,W,H).unsqueeze(1) 
        return shading



# a mixer (linear layer)
class waspShadeRenderer2(nn.Module):
    def __init__(self, opt):
        super(waspShadeRenderer2, self).__init__()
        self.opt = opt
        self.getHomo = HomogeneousCoord(opt)
        self.getMMatrix = MMatrix(opt)
    def forward(self, light, normals):
        # homogeneous coordinate of the normals
        batchSize = normals.size(0)
        W = normals.size(2)
        H = normals.size(3)
        hNormals = self.getHomo(normals)
        # matrix for light
        mLight = self.getMMatrix(light)
        # get shading from these two: N x 4 , N = batchSize x W x H 
        hN_vec = hNormals.view(batchSize, 4, -1).permute(0,2,1).contiguous().view(-1,4)
        # N x 1 x 4
        hN_vec_Left  = hN_vec.unsqueeze(1)
        # N x 4 x 1
        hN_vec_Right = hN_vec.unsqueeze(2)
        # expand the lighting from batchSize x 4 x 4 to N x 4 x 4
        hL = mLight.view(batchSize,16).repeat(1,W*H).t().contiguous().view(-1,4,4)
        shade0 = torch.matmul(hN_vec_Left, hL)
        shade1 = torch.matmul(shade0, hN_vec_Right)
        #shade1 is tensor of size Nx1x1 = batchSize x W x H
        shading = shade1.view(batchSize,W,H).unsqueeze(1) 
        return shading


# The decoders that use residule warper
class Dense_DecodersIntegralWarper2_Light(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Light, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.ndim = opt.ndim
        self.ldim = opt.ldim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        # get the normal direction phi, theta, range in [0 1]
        self.decoderN_angles = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        # convert angles phi, theta, [0,1] to [0, pi] to [nx,ny,nz]
        self.decoderN_vectors = waspAngleToDirection(opt)
        self.renderShading = waspShadeRenderer(opt)
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
        self.shadeCutter = nn.Hardtanh(0.001,1)
        self.getNormalImage = getNormalImage(opt)
    def forward(self, zL, zN, zT, zW, basegrid):
        #self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.light = zL
        self.normals_angles = self.decoderN_angles(zN.view(-1,self.ndim,1,1))
        self.normals = self.decoderN_vectors(self.normals_angles)
        self.Inormals = self.getNormalImage(self.normals)
        self.shading = self.renderShading(self.light, self.normals)
        self.shading = self.shadeCutter(self.shading)
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.normals, self.shading, self.texture, self.img, self.resWarping, self.output, self.warping, self.Inormals


# The encoders
class Dense_Encoders_Light2(nn.Module):
    def __init__(self, opt):
        super(Dense_Encoders_Light2, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspDenseEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        #self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zLmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.ldim)
        self.zNmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.ndim)
        self.zTmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.tdim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        #self.zImg  = self.zImixer(self.z)
        self.zLight = self.zLmixer(self.z)
        self.zNormals = self.zNmixer(self.z)
        self.zTexture = self.zTmixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zLight, self.zNormals, self.zTexture, self.zWarp


# The decoders that use residule warper
class Dense_DecodersIntegralWarper2_Light2(nn.Module):
    def __init__(self, opt):
        super(Dense_DecodersIntegralWarper2_Light2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.ndim = opt.ndim
        self.ldim = opt.ldim
        self.sdim = opt.sdim
        self.tdim = opt.tdim
        self.wdim = opt.wdim
        self.decoderL = waspMixerHardtanh(opt, ngpu=1, nin = opt.ldim, nout = 9)
        # get the normal direction phi, theta, range in [0 1]
        self.decoderN_angles = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.sdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        # convert angles phi, theta, [0,1] to [0, pi] to [nx,ny,nz]
        self.decoderN_vectors = waspAngleToDirection(opt)
        self.renderShading = waspShadeRenderer(opt)
        self.decoderT = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.tdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDenseDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1, activation=nn.Tanh, args=[], f_activation=nn.Sigmoid, f_args=[])
        self.intrinsicComposer = waspIntrinsicComposer(opt)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
        self.shadeCutter = nn.Hardtanh(0.001,1)
        self.getNormalImage = getNormalImage(opt)

    def forward(self, zL, zN, zT, zW, basegrid):
        #self.shading = self.decoderS(zS.view(-1,self.sdim,1,1))
        self.light = self.decoderL(zL)
        self.light = self.light*8
        self.normals_angles = self.decoderN_angles(zN.view(-1,self.ndim,1,1))
        self.normals = self.decoderN_vectors(self.normals_angles)
        self.Inormals = self.getNormalImage(self.normals)
        self.shading = self.renderShading(self.light, self.normals)
        self.shading = self.shadeCutter(self.shading)
        self.texture = self.decoderT(zT.view(-1,self.tdim,1,1))
        self.img = self.intrinsicComposer(self.shading, self.texture)
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.img, self.warping)
        return self.normals, self.shading, self.texture, self.img, self.resWarping, self.output, self.warping, self.Inormals, self.light



class DeformationApply(nn.Module):
    def __init__(self, opt):
        super(DeformationApply, self).__init__()
        self.warper   = waspWarper(opt)
    def forward(self, image, warping):
        output  = self.warper(image, warping)
        return output



# an encoder architecture
class SHClassifier(nn.Module):
    def __init__(self, opt, ngpu=1, ndim = 9):
        super(SHClassifier, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(self.ndim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, input):
        output = self.main(input)
        #print(output.size())
        return output   


class PatchGANDiscriminator(nn.Module):
    def __init__(self, opt, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PatchGANDiscriminator, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
##