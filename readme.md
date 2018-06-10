# What’s this?
Implementing **DCGAN**(Radford et al. (2015)(https://arxiv.org/pdf/1511.06434.pdf)) by using MNIST dataset


## Network structure(Generator)

![imgur](https://i.imgur.com/zz8CuoI.png "Generatorの図解")  

## Train loop

以下の繰り返し(epoch数はxのtrain sample数に依存)
1. Discriminatorの学習
  1. batch_size個だけzを取り出す
  1. batch_size個だけxを取り出す
  1. Dを更新する
2. Generatorの学習
  1. batch_size個だけG(z)を生成
  1. Gを更新

# Setting of Hyper parameter, etc
## Common in Discriminator and Generator
* Batch sizeは128
* Adamを使う
* learning late(Adamのalpha)は0.0002
* Adamは`beta1 = 0.5`
* 前処理なし
* Batch Normalizationは平均ゼロ，分散1の乱数，Gの出力及びDの入力には使わない
* 重みの初期値はすべて平均ゼロ，分散0.02の正規分布に従う乱数


## Discriminator
* poolingを使わず，strided convolutionsのみ(普通のConvolution)
* 全結合(Fully connected)を深い層で使わない
* LeakyReLU(傾き0.2)を全レイヤで使う

## Generator
- pooling → fractional-strided convolutions(ChainerでいうDeconvolution)
- 全結合を深い層では使わない
- 最終層はtanhを，それ以外はReLUを使う

## Reference
- *mattya/chainer-DCGAN: Chainer implementation of Deep Convolutional Generative Adversarial Network*
https://github.com/mattya/chainer-DCGAN
- *できるだけ丁寧にGANとDCGANを理解する*
http://mizti.hatenablog.com/entry/2016/12/10/224426
***
**To do**
* MNIST以外の様々な画像に対応出来るものを
