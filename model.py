import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer.functions import caffe
from chainer import Variable, optimizers
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import mean_squared_error
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
import math
import numpy as np

class ResidualBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(n_in, n_out, ksize, stride, 1, w),
            c2=L.Convolution2D(n_out, n_out, ksize, 1, 1, w),
            b1=L.BatchNormalization(n_out),
            b2=L.BatchNormalization(n_out)
        )

    def __call__(self, x, test):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = self.b2(self.c2(h), test=test)
        """
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        """
        return h + x

class CNNAE2ResNet(chainer.Chain):

    def __init__(self, train=True):
        w = chainer.initializers.Normal(0.02)
        super(CNNAE2ResNet, self).__init__(
            c0 = L.Convolution2D(None, 64, 4, stride=2, pad=1, initialW=w), # 1024 -> 512
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w),  # 512 -> 256
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w), # 256 -> 128
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w), # 128 -> 64
            c4 = L.Convolution2D(512, 512, 4, stride=2, pad=1, initialW=w), # 64 -> 32
            c5 = L.Convolution2D(512, 512, 4, stride=2, pad=1, initialW=w), # 32 -> 16

            ra = ResidualBlock(512, 512),
            rb = ResidualBlock(512, 512),

            dc0a = L.Deconvolution2D(512, 512, 4, stride=2, pad=1, initialW=w),
            dc1a = L.Deconvolution2D(1024, 512, 4, stride=2, pad=1, initialW=w),
            dc2a = L.Deconvolution2D(1024, 256, 4, stride=2, pad=1, initialW=w),
            dc3a = L.Deconvolution2D(512, 128, 4, stride=2, pad=1, initialW=w),
            dc4a = L.Deconvolution2D(256, 64, 4, stride=2, pad=1, initialW=w),
            dc5a = L.Deconvolution2D(128, 9, 4, stride=2, pad=1, initialW=w),

            dc0b = L.Deconvolution2D(512, 512, 4, stride=2, pad=1, initialW=w),
            dc1b = L.Deconvolution2D(1024, 512, 4, stride=2, pad=1, initialW=w),
            dc2b = L.Deconvolution2D(1024, 256, 4, stride=2, pad=1, initialW=w),
            dc3b = L.Deconvolution2D(512, 128, 4, stride=2, pad=1, initialW=w),
            dc4b = L.Deconvolution2D(256, 64, 4, stride=2, pad=1, initialW=w),
            dc5b = L.Deconvolution2D(128, 3, 4, stride=2, pad=1, initialW=w),

            c0l = L.Convolution2D(None, 512, 4, stride=2, pad=1, initialW=w), # 16 -> 8
            c1l = L.Convolution2D(512, 256, 4, stride=2, pad=1, initialW=w),  # 8 -> 4
            c2l = L.Convolution2D(256, 128, 4, stride=2, pad=1, initialW=w), # 4 -> 2
            c3l = L.Convolution2D(128, 27, 4, stride=2, pad=1, initialW=w), # 2 -> 1
            
            bnc1 = L.BatchNormalization(128),
            bnc2 = L.BatchNormalization(256),
            bnc3 = L.BatchNormalization(512),
            bnc4 = L.BatchNormalization(512),
            bnc5 = L.BatchNormalization(512),

            bndc0a = L.BatchNormalization(512),
            bndc1a = L.BatchNormalization(512),
            bndc2a = L.BatchNormalization(256),
            bndc3a = L.BatchNormalization(128),
            bndc4a = L.BatchNormalization(64),

            bndc0b = L.BatchNormalization(512),
            bndc1b = L.BatchNormalization(512),
            bndc2b = L.BatchNormalization(256),
            bndc3b = L.BatchNormalization(128),
            bndc4b = L.BatchNormalization(64),
            
            bnc0l = L.BatchNormalization(512),
            bnc1l = L.BatchNormalization(256),
            bnc2l = L.BatchNormalization(128)
        )
        self.train = train
        self.train_dropout = train

    def __call__(self, xi):
        hc0 = F.leaky_relu(self.c0(xi))
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0), test=not self.train))
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1), test=not self.train))
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2), test=not self.train))
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3), test=not self.train))
        hc5 = F.leaky_relu(self.bnc5(self.c5(hc4), test=not self.train))

        hra = self.ra(hc5, test=not self.train)

        ha = F.relu(F.dropout(self.bndc0a(self.dc0a(hra), test=not self.train), 0.5, train=self.train_dropout))
        ha = F.concat((ha,hc4))
        ha = F.relu(F.dropout(self.bndc1a(self.dc1a(ha), test=not self.train), 0.5, train=self.train_dropout))
        ha = F.concat((ha,hc3))
        ha = F.relu(F.dropout(self.bndc2a(self.dc2a(ha), test=not self.train), 0.5, train=self.train_dropout))
        ha = F.concat((ha,hc2))
        ha = F.relu(self.bndc3a(self.dc3a(ha), test=not self.train))
        ha = F.concat((ha,hc1))
        ha = F.relu(self.bndc4a(self.dc4a(ha), test=not self.train))
        ha = F.concat((ha,hc0))
        ha = self.dc5a(ha)

        hrb = self.rb(hc5, test=not self.train)
        
        hb = F.relu(F.dropout(self.bndc0b(self.dc0b(hrb), test=not self.train), 0.5, train=self.train_dropout))
        hb = F.concat((hb,hc4))
        hb = F.relu(F.dropout(self.bndc1b(self.dc1b(hb), test=not self.train), 0.5, train=self.train_dropout))
        hb = F.concat((hb,hc3))
        hb = F.relu(F.dropout(self.bndc2b(self.dc2b(hb), test=not self.train), 0.5, train=self.train_dropout))
        hb = F.concat((hb,hc2))
        hb = F.relu(self.bndc3b(self.dc3b(hb), test=not self.train))
        hb = F.concat((hb,hc1))
        hb = F.relu(self.bndc4b(self.dc4b(hb), test=not self.train))
        hb = F.concat((hb,hc0))
        hb = F.clip(self.dc5b(hb), 0.0, 1.0)
        
        hc = F.concat((hc5, hra, hrb))
        
        hc = F.leaky_relu(self.bnc0l(self.c0l(hc), test=not self.train))
        hc = F.leaky_relu(self.bnc1l(self.c1l(hc), test=not self.train))
        hc = F.leaky_relu(self.bnc2l(self.c2l(hc), test=not self.train))
        hc = F.reshape(self.c3l(hc), (9, 3))

        return ha, hb, hc
