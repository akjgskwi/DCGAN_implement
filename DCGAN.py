# -*- coding: utf-8 -*-

import numpy as np


from chainer.datasets import mnist
from chainer import Chain, Variable, optimizers, serializers, initializers, iterators
import chainer.functions as F
import chainer.links as L


z_size = 100
batch_size = 128
lrelu = F.leaky_relu
max_epoch = 20
# networkの定義


class Generator(Chain):
    """Generate a Pic of a digit from a random vector z."""

    def __init__(self, initializer=initializers.Normal(scale=0.02)):
        super(Generator, self).__init__(
            linear=L.Linear(z_size, out_size=2 * 2 *
                            256, initialW=initializer),
            deconv1=L.Deconvolution2D(in_channels=256, out_channels=128, pad=0,
                                      stride=1, ksize=2, outsize=(3, 3), initialW=initializer),
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
        # reshape? transpose?
        # h = F.reshape(F.relu(self.bn0(self.linear(z))),
        #              (z.data.shape[0], 256, 2, 2))
        h = F.reshape(F.relu(self.bn0(self.linear(z))),
                      (z.data.shape[0], 256, 2, 2))
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        # use Tanh as activation function only for the last layer
        x = F.tanh(self.deconv4(h))
        return x


class Discriminator(Chain):
    """Read the digits and investigate
    these are made by Generator or from sample dataset."""

    def __init__(self, initializer=initializers.Normal(scale=0.02)):
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(in_channels=1, out_channels=32, pad=1,
                                  stride=2, ksize=4, initialW=initializer),
            conv2=L.Convolution2D(in_channels=32, out_channels=64, pad=1,
                                  stride=2, ksize=4, initialW=initializer),
            conv3=L.Convolution2D(in_channels=64, out_channels=128, pad=0,
                                  stride=2, ksize=3, initialW=initializer),
            conv4=L.Convolution2D(in_channels=128, out_channels=256, pad=0,
                                  stride=1, ksize=2, initialW=initializer),
            linear=L.Linear(2 * 2 * 256, 2, initialW=initializer),
            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
        )

    def __call__(self, x):
        h = lrelu(self.bn1(self.conv1(x)))
        h = lrelu(self.bn2(self.conv2(h)))
        h = lrelu(self.bn3(self.conv3(h)))
        h = lrelu(self.bn4(self.conv4(h)))
        y = self.linear(h)
        return y


if __name__ == "__main__":
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
    ans_mnist = Variable(np.ones(batch_size, dtype=np.int32))
    ans_gen = Variable(np.zeros(batch_size, dtype=np.int32))

    # Train loop
    # while number of train is sufficient
    # train D and G by pic from G

    train, test = mnist.get_mnist(withlabel=False, ndim=3)
    mnist_iter = iterators.SerialIterator(train, batch_size)


    while mnist_iter.epoch < max_epoch:
        print("epoch: ", mnist_iter.epoch, "iterations: ", mnist_iter.current_position)
        z = Variable(np.random.uniform(-1.0, 1.0,
                                       (batch_size, z_size)).astype(dtype=np.float32))
        x = Gen(z)
        y_G = Dis(x)
        loss_D = F.softmax_cross_entropy(y_G, ans_gen)
        loss_G = F.softmax_cross_entropy(y_G, ans_mnist)

        mnist_batch = mnist_iter.next()
        y_mnist = Dis(np.array(mnist_batch, dtype=np.float32))
        loss_D += F.softmax_cross_entropy(y_mnist, ans_mnist)

        Dis.cleargrads()
        loss_D.backward()
        optimizer_D.update()

        Gen.cleargrads()
        loss_G.backward()
        optimizer_G.update()

        if mnist_iter.is_new_epoch:
            serializers.save_npz("Generator_epoch%d.npz" % mnist_iter.epoch, Gen)

    serializers.save_npz('Generator_final.npz', Gen)
