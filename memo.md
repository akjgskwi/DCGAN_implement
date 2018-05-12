# 論文([Radford et al. (2015)](https://arxiv.org/pdf/1511.06434.pdf)の実装設定メモ


***やったこと***
***
 * D,Gの構造,呼び出し
***
***やること***
***
 * loss関数の作成
 * 学習ループ(Algorithmに沿って)
***

***学習ループ***
以下の繰り返し(訓練回数ぶん)
1. Discriminatorの学習
  1. batch_size個だけG(z)を生成する．
  2. batch_size個だけx[i]を取り出し，D(x)を計算する
  3. Dを更新する
2. Generatorの学習
  1. batch_size個だけG(z)を生成
  2. Gを更新


***構造上の注意***
***
![imgur](https://i.imgur.com/zz8CuoI.png "Generatorの図解")  

***共通***
  * Batch sizeは128
  * Adamを使う
  * learning late(Adamのalpha)は0.0002
  * Adamは`beta1 = 0.5`
  * 前処理なし
  * Batch正規化は平均ゼロ，分散1の乱数，Gの出力とDの入力には使わない
  * 重みはすべて平均ゼロ，分散0.02,


***Discriminator***
  * pooling → strided convolutions(普通のConvolution)
  * 全結合(Fully connected)を深い層で使わない
  * LeakyReLU(傾き0.2)を全レイヤで使う

***Generator***
  - pooling → fractional-strided convolutions(ChainerではDeconvolution)
  - 全結合を深い層で使わない
  - 最終層はtanhを，それ以外はReLUを使う

***

***訓練の詳細***

***
  -
