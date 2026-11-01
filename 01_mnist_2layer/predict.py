import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from functions import softmax

# 1. テストデータの読み込み（正解ラベルは 0〜9 の数字のままにする）
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

# 2. ネットワークを作り、学習済みパラメータを読み込む
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
params = np.load("params.npz")
for key in ('W1', 'b1', 'W2', 'b2'):
    # [...] = で「中身だけ」を書き換える。
    # network.params[key] = params[key] とすると params 側の名札だけが付け替わり、
    # レイヤ（Affine）が見ている W, b はランダムな初期値のままになってしまう。
    network.params[key][...] = params[key]

# 3. テストデータ全体で推論
scores = network.predict(x_test)          # (10000, 10) のスコア
probs = softmax(scores)                    # 確率に変換
pred = np.argmax(probs, axis=1)            # 一番確率が高いクラス
conf = np.max(probs, axis=1)               # そのときの確率（自信度）

accuracy = np.mean(pred == t_test)
print(f"テスト精度: {accuracy:.4f}  ({np.sum(pred == t_test)} / {len(t_test)})")

wrong_idx = np.where(pred != t_test)[0]
print(f"間違えた枚数: {len(wrong_idx)}")


def show_images(indices, title):
    """指定した番号の画像を、予測・正解・自信度つきで並べて表示する"""
    n = len(indices)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, i in zip(axes, indices):
        ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
        ok = pred[i] == t_test[i]
        ax.set_title(f"pred {pred[i]} ({conf[i]:.0%})\nlabel {t_test[i]}",
                     color='black' if ok else 'red', fontsize=10)
    fig.suptitle(title)
    plt.tight_layout()


# 4. ランダムに選んだ10枚
random_idx = np.random.choice(len(x_test), 10, replace=False)
show_images(random_idx, "Random test images")

# 5. 間違えた画像のうち、自信満々で間違えたもの上位10枚
worst_idx = wrong_idx[np.argsort(-conf[wrong_idx])][:10]
show_images(worst_idx, "Most confident mistakes")

plt.show()