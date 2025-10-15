import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import GroupKFold

# Irisデータセットの読み込み
iris = load_iris()
X, y = iris.data, iris.target

# グループを作成（例：各クラスをさらに細かいグループに分ける）
# Irisは各クラス50サンプルずつなので、10個のグループを作成
# グループサイズを変えて、GroupKFoldの振る舞いを確認
groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 0-9
                   2, 2, 2, 2, 2, 3, 3, 3, 3, 3,  # 10-19
                   4, 4, 4, 4, 4, 5, 5, 5, 5, 5,  # 20-29
                   6, 6, 6, 6, 6, 7, 7, 7, 7, 7,  # 30-39
                   8, 8, 8, 8, 8, 9, 9, 9, 9, 9,  # 40-49
                   # クラス1: グループサイズを変える（偏りを作る）
                   10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  # 50-59 (10サンプル)
                   11, 11, 11, 11, 11, 11, 11, 11, 11, 11,  # 60-69 (10サンプル)
                   12, 12, 12, 12, 12, 13, 13, 13, 13, 13,  # 70-79
                   14, 14, 14, 14, 14, 15, 15, 15, 15, 15,  # 80-89
                   16, 16, 16, 16, 16, 17, 17, 17, 17, 17,  # 90-99
                   # クラス2
                   18, 18, 18, 18, 18, 19, 19, 19, 19, 19,  # 100-109
                   20, 20, 20, 20, 20, 21, 21, 21, 21, 21,  # 110-119
                   22, 22, 22, 22, 22, 23, 23, 23, 23, 23,  # 120-129
                   24, 24, 24, 24, 24, 25, 25, 25, 25, 25,  # 130-139
                   26, 26, 26, 26, 26, 27, 27, 27, 27, 27])  # 140-149

print("データセット情報:")
print(f"総サンプル数: {len(X)}")
print(f"ユニークなグループ数: {len(np.unique(groups))}")
print(f"各グループのサンプル数: {np.bincount(groups)}")
print()

# GroupKFoldの設定（5分割、shuffle=False）
print("=" * 70)
print("GroupKFold (shuffle=False) - サンプル数でバランスされた分割")
print("=" * 70)
gkf_no_shuffle = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf_no_shuffle.split(X, y, groups), 1):
    test_groups = np.unique(groups[test_idx])
    print(f"\nFold {fold}:")
    print(f"  訓練サンプル数: {len(train_idx)}, テストサンプル数: {len(test_idx)}")
    print(f"  テストグループ: {test_groups}")
    print(f"  テストグループのサンプル数: {[np.sum(groups == g) for g in test_groups]}")

# GroupKFoldの設定（5分割、shuffle=True）
print("\n" + "=" * 70)
print("GroupKFold (shuffle=True) - ランダムにグループを分割")
print("=" * 70)
gkf_shuffle = GroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(gkf_shuffle.split(X, y, groups), 1):
    test_groups = np.unique(groups[test_idx])
    print(f"\nFold {fold}:")
    print(f"  訓練サンプル数: {len(train_idx)}, テストサンプル数: {len(test_idx)}")
    print(f"  テストグループ: {test_groups}")
    print(f"  テストグループのサンプル数: {[np.sum(groups == g) for g in test_groups]}")

# ビジュアライゼーション: shuffle=False vs shuffle=True
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# shuffle=False
n_samples = len(X)
n_splits = 5
fold_data_no_shuffle = np.zeros((n_splits, n_samples))

for fold, (train_idx, test_idx) in enumerate(gkf_no_shuffle.split(X, y, groups)):
    fold_data_no_shuffle[fold, train_idx] = 1
    fold_data_no_shuffle[fold, test_idx] = 2

cmap = plt.cm.colors.ListedColormap(['white', 'skyblue', 'coral'])
bounds = [0, 1, 2, 3]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im1 = ax1.imshow(fold_data_no_shuffle, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('Fold Number', fontsize=12)
ax1.set_title('GroupKFold (shuffle=False) - Balanced by Sample Count', fontsize=14, pad=20)
ax1.set_yticks(range(n_splits))
ax1.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
ax1.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

# shuffle=True
fold_data_shuffle = np.zeros((n_splits, n_samples))

for fold, (train_idx, test_idx) in enumerate(gkf_shuffle.split(X, y, groups)):
    fold_data_shuffle[fold, train_idx] = 1
    fold_data_shuffle[fold, test_idx] = 2

im2 = ax2.imshow(fold_data_shuffle, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
ax2.set_xlabel('Sample Index', fontsize=12)
ax2.set_ylabel('Fold Number', fontsize=12)
ax2.set_title('GroupKFold (shuffle=True) - Random Group Assignment', fontsize=14, pad=20)
ax2.set_yticks(range(n_splits))
ax2.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
ax2.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

# カラーバーの追加
cbar = plt.colorbar(im2, ax=[ax1, ax2], ticks=[0.5, 1.5, 2.5])
cbar.ax.set_yticklabels(['Unused', 'Train', 'Test'])

plt.tight_layout()
plt.savefig('groupkfold_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved as 'groupkfold_comparison.png'")
plt.show()
