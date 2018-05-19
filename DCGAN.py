# -*- coding: utf-8 -*-

import numpy as np

import chainer
from chainer.datasets import mnist
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, initializers, iterators
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


z_size = 100
batch_size = 128
LReLU = F.LeakyReLU(slope=0.2)


# networkの定義


class Generator(Chain):
    """Generate a Pic of a digit from a random vector z."""

    def __init__(self, initializer=initializers.Normal(scale=0.02)):
        super(Generator, self).__init__(
            linear=L.Linear(100, out_size=2 * 2 * 256, initialW=initializer),
            deconv1=L.Deconvolution2D(in_channels=256, out_channels=128, pad=1,
                                      stride=1, ksize=3, outsize=(3, 3), initialW=initializer),
            deconv2=L.Deconvolution2D(in_channels=128, out_channels=64, pad=0,
                                      stride=2, ksize=3, outsize=(7, 7), initialW=initializer),
            deconv3=L.Deconvolution2D(in_channels=64, out_channels=32, pad=1,
                                      stride=2, ksize=4, outsize=(14, 14), initialW=initializer),
            deconv4=L.Deconvolution2D(in_channels=32, out_channels=1, pad=1,
                                      stride=2, ksize=4, outsize=(28, 28), initialW=initializer),
            bn0=L.BatchNormalization(2 * 2 * 256),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(32),
        )

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.linear(z))),
                      (z.data.shape[0], 256, 2, 2))
        h = F.ReLU(self.bn1(self.deconv1(h)))
        h = F.ReLU(self.bn2(self.deconv2(h)))
        h = F.ReLU(self.bn3(self.deconv3(h)))
        # use Tanh as activation function only for the last layer
        x = F.Tanh(self.deconv4(h))
        return x


class Discriminator(Chain):
    """Read the digits and investigate these are made by Generator or from sample dataset."""

    def __init__(self, initializer=initializers.Normal(scale=0.02)):
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(in_channels=1, out_channels=32, pad=1,
                                  stride=2, ksize=4, outsize=(14, 14), initialW=initializer),
            conv2=L.Convolution2D(in_channels=32, out_channels=64, pad=1,
                                  stride=2, ksize=4, outsize=(7, 7), initialW=initializer),
            conv3=L.Convolution2D(in_channels=64, out_channel=128, pad=0,
                                  stride=2, ksize=3, outsize=(3, 3), initialW=initializer),
            conv4=L.Convolution2D(in_channels=128, out_channels=256, pad=1,
                                  stride=1, ksize=3, outsize=(2, 2), initialW=initializer),
            linear=L.Linear(2 * 2 * 256, 2, initialW=initializer),
            bn1=L.BatchNormalization(14 * 14 * 32),
            bn2=L.BatchNormalization(7 * 7 * 64),
            bn3=L.BatchNormalization(3 * 3 * 128),
            bn4=L.BatchNormalization(2 * 2 * 256),
        )

    def __call__(self, x):
        h = LReLU(self.bn1(self.conv1(x)))
        h = LReLU(self.bn2(self.conv2(h)))
        h = LReLU(self.bn3(self.conv3(h)))
        h = LReLU(self.bn4(self.conv4(h)))
        y = self.linear(h)
        return y


gpu_id = -1  # gpu_id = 0 if using GPU

Gen = Generator()
Dis = Discriminator()
if gpu_id >= 0:
    Gen.to_gpu(gpu_id)
    Dis.to_gpu(gpu_id)


optimizer_G = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_D = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_G.setup(Gen)
optimizer_D.setup(Dis)

# 1:pic from dataset 0:pic from Generator
ans_mnist = np.ones(batch_size, dtype=np.int32)
ans_gen = np.zeros(batch_size, dtype=np.int32)


# Train loop
# while number of train is sufficient
# train D and G by pic from G

train, test = mnist.get_mnist(withlabel=False)
mnist_iter = iterators.SerialIterator(train, batch_size)
max_epoch = 10

while mnist_iter.epoch < max_epoch:
    z = Variable(np.random.uniform(-1.0, 1.0, (batch_size, z_size)))
    x = Gen(z)
    y_G = Dis(x)
    loss_D = F.softmax_cross_entropy(y_G, ans_gen)
    loss_G = F.softmax_cross_entropy(y_G, ans_mnist)

    mnist_batch = mnist_iter.next()
    y_mnist = Dis(mnist_batch)
    loss_D += F.softmax_cross_entropy(y_mnist, ans_mnist)

    Gen.cleargrads()
    loss_G.backward()
    optimizer_G.update()

    Dis.cleargrads()
    loss_D.backward()
    optimizer_D.update()


serializers.save_npz('Generator.model', Gen)
