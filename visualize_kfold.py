import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

# Irisデータセットの読み込み
iris = load_iris()
X, y = iris.data, iris.target

# KFoldの設定（5分割）
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ビジュアライゼーション用の準備
n_samples = len(X)
n_splits = 5

# プロット用の配列を作成
fold_data = np.zeros((n_splits, n_samples))

# 各foldの分割情報を記録
for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    fold_data[fold, train_idx] = 1  # 訓練データ
    fold_data[fold, test_idx] = 2  # テストデータ

    print(f"Fold {fold + 1}:")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    print()

# ビジュアライゼーション
fig, ax = plt.subplots(figsize=(15, 6))

# カスタムカラーマップ
cmap = plt.cm.colors.ListedColormap(["white", "skyblue", "coral"])
bounds = [0, 1, 2, 3]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# ヒートマップとして表示
im = ax.imshow(fold_data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

# 軸のラベル設定
ax.set_xlabel("Sample Index", fontsize=12)
ax.set_ylabel("Fold Number", fontsize=12)
ax.set_title("KFold Cross-Validation Split Pattern (Iris Dataset)", fontsize=14, pad=20)

# Y軸の設定
ax.set_yticks(range(n_splits))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(n_splits)])

# カラーバーの追加
cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5])
cbar.ax.set_yticklabels(["Unused", "Train", "Test"])

# グリッドの追加
ax.set_xticks(np.arange(0, n_samples, 10), minor=True)
ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig("kfold_visualization.png", dpi=150, bbox_inches="tight")
print("Visualization saved as 'kfold_visualization.png'")
