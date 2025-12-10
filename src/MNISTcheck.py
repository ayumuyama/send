import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# MNISTデータのロード（tensorとして取得）
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 1枚画像を取り出す
image, label = mnist_train[0]  # image.shape: [1, 28, 28]

# スパイク符号化（Poissonノイズによる）
dt = 1e-3  # 時間刻み
tuning = 0.035  # 発火率（最大）
rate = image / image.max() * tuning  # ピクセル強度に応じた発火率
spike_prob = rate / dt

# スパイク生成
rand = torch.rand_like(image)
input_spikes = (rand < spike_prob).float()

# 表示
plt.imshow(torch.sum(input_spikes, dim=0).squeeze().numpy(), cmap='gray')
plt.title(f"Label: {label}")
plt.savefig('results/MNISTcheck.png')