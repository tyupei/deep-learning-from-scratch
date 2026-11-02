# deep-learning-from-scratch

『ゼロから作るDeep Learning ― Pythonで学ぶディープラーニングの理論と実装』（斎藤康毅 著、オライリー・ジャパン）を学習しながら、ニューラルネットワークを **NumPy だけで** 一から実装していくリポジトリです。

本のコードを写すのではなく、各レイヤの順伝播・逆伝播を自分で導出・実装し、数値微分による勾配確認や推論結果の分析まで行うことを目標にしています。

## 進捗

| # | 内容 | 主なトピック | 結果 |
|---|---|---|---|
| [01](./01_mnist_2layer) | MNIST 手書き数字分類（2層ニューラルネット） | Affine / ReLU / Softmax-with-Loss レイヤ、誤差逆伝播法、勾配確認 | テスト精度 約97% |
| 02 | 学習に関するテクニック（予定） | SGD / Momentum / AdaGrad / Adam、重みの初期値、Batch Normalization | – |
| 03 | 畳み込みニューラルネットワーク（予定） | Convolution / Pooling レイヤ | – |

## 動作環境

- Python 3
- NumPy
- Matplotlib

ディープラーニング用のフレームワーク（PyTorch、TensorFlow など）は使用していません。

## 参考文献

- 斎藤康毅『ゼロから作るDeep Learning ― Pythonで学ぶディープラーニングの理論と実装』オライリー・ジャパン, 2016.
- 公式リポジトリ: https://github.com/oreilly-japan/deep-learning-from-scratch

`dataset/mnist.py`（MNIST の読み込み）は上記の公式リポジトリのものを使用しています。それ以外のコードは自分で実装したものです。
