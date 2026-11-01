import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2. ネットワークの作成
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3. ハイパーパラメータ
iters_num = 10000       # 更新の回数
batch_size = 100        # ミニバッチの大きさ
learning_rate = 0.1     # 学習率

train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)   # 1エポックあたりの更新回数（600回）

# 記録用のリスト
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 4. 学習ループ
for i in range(iters_num):
    # a. ミニバッチをランダムに選ぶ
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # b. 誤差逆伝播法で勾配を求める
    grad = network.gradient(x_batch, t_batch)

    # c. パラメータの更新（-= でその場で書き換える。レイヤ側の W, b も一緒に変わる）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # d. 損失の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # e. 1エポックごとに精度を記録
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"epoch {len(train_acc_list) - 1:2d} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

# 5. 学習済みパラメータの保存（predict.py で使う）
np.savez("params.npz", **network.params)

# 6. グラフの描画
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 左：損失の推移
ax1.plot(train_loss_list, linewidth=0.5)
ax1.set_xlabel("iteration")
ax1.set_ylabel("loss")
ax1.set_title("Training loss")

# 右：精度の推移
epochs = np.arange(len(train_acc_list))
ax2.plot(epochs, train_acc_list, marker='o', label='train acc')
ax2.plot(epochs, test_acc_list, marker='s', linestyle='--', label='test acc')
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.set_ylim(0, 1.0)
ax2.set_title("Accuracy")
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()