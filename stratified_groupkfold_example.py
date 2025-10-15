"""
StratifiedGroupKFoldの動作を確認するサンプルプログラム

StratifiedGroupKFoldは以下の制約を満たすようにデータを分割します：
1. 同じグループのサンプルは同じfoldに入る（GroupKFoldと同様）
2. 各foldでクラス比率を可能な限り保持する（StratifiedKFoldと同様）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from generate_imbalanced_data import generate_imbalanced_iris_like_data, print_dataset_info


def analyze_fold_distribution(
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_num: int
) -> None:
    """
    各フォールドの分布を分析して表示

    Parameters
    ----------
    y : np.ndarray
        クラスラベル
    groups : np.ndarray
        グループラベル
    train_idx : np.ndarray
        訓練データのインデックス
    test_idx : np.ndarray
        テストデータのインデックス
    fold_num : int
        フォールド番号
    """
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    print(f"\n{'='*80}")
    print(f"Fold {fold_num}")
    print(f"{'='*80}")
    print(f"訓練データ数: {len(train_idx)}, テストデータ数: {len(test_idx)}")

    # クラス分布
    print(f"\n各クラスの分布:")
    print(f"{'クラス':<10} {'訓練':<20} {'テスト':<20} {'全体':<20}")
    print(f"{'-'*80}")

    n_classes = len(np.unique(y))
    for class_idx in range(n_classes):
        train_count = np.sum(y_train == class_idx)
        test_count = np.sum(y_test == class_idx)
        total_count = np.sum(y == class_idx)

        train_ratio = train_count / len(train_idx) * 100
        test_ratio = test_count / len(test_idx) * 100
        total_ratio = total_count / len(y) * 100

        print(f"クラス {class_idx:<4} "
              f"{train_count:4d} ({train_ratio:5.1f}%)      "
              f"{test_count:4d} ({test_ratio:5.1f}%)      "
              f"{total_count:4d} ({total_ratio:5.1f}%)")

    # グループ分布
    print(f"\nグループ分布:")
    print(f"  訓練グループ数: {len(np.unique(groups_train))}")
    print(f"  テストグループ数: {len(np.unique(groups_test))}")
    print(f"  テストグループID: {sorted(np.unique(groups_test))}")

    # 重要: 同じグループが訓練とテストの両方に存在しないことを確認
    train_groups_set = set(groups_train)
    test_groups_set = set(groups_test)
    overlap = train_groups_set & test_groups_set

    if overlap:
        print(f"  ⚠️  警告: 訓練とテストで重複するグループ: {overlap}")
    else:
        print(f"  ✓ グループの重複なし（正しく分割されている）")

    # テストグループの詳細
    print(f"\nテストグループの詳細:")
    for group_id in sorted(np.unique(groups_test)):
        group_mask = groups_test == group_id
        group_size = np.sum(group_mask)
        group_y = y_test[group_mask]
        group_class_dist = np.bincount(group_y, minlength=n_classes)
        print(f"  グループ {group_id:2d}: {group_size}サンプル (クラス分布: {group_class_dist})")


def visualize_stratified_groupkfold(
    y: np.ndarray,
    groups: np.ndarray,
    sgkf: StratifiedGroupKFold,
    X: np.ndarray
) -> None:
    """
    StratifiedGroupKFoldの分割パターンを可視化

    Parameters
    ----------
    y : np.ndarray
        クラスラベル
    groups : np.ndarray
        グループラベル
    sgkf : StratifiedGroupKFold
        StratifiedGroupKFoldオブジェクト
    X : np.ndarray
        特徴量行列（splitメソッドに必要）
    """
    n_splits = sgkf.n_splits
    n_samples = len(y)
    n_classes = len(np.unique(y))

    # 図の準備
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.3, wspace=0.3)

    # 1. Fold分割のヒートマップ
    ax1 = fig.add_subplot(gs[0, :])
    fold_data = np.zeros((n_splits, n_samples))

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_data[fold, train_idx] = 1  # Train
        fold_data[fold, test_idx] = 2   # Test

    cmap = plt.cm.colors.ListedColormap(['white', 'skyblue', 'coral'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im1 = ax1.imshow(fold_data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Fold Number', fontsize=12)
    ax1.set_title('StratifiedGroupKFold Split Pattern', fontsize=14, pad=20)
    ax1.set_yticks(range(n_splits))
    ax1.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])

    # グループの境界線を追加
    group_boundaries = []
    current_group = groups[0]
    for i in range(1, len(groups)):
        if groups[i] != current_group:
            group_boundaries.append(i - 0.5)
            current_group = groups[i]

    for boundary in group_boundaries:
        ax1.axvline(x=boundary, color='black', linewidth=1, alpha=0.3, linestyle='--')

    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0.5, 1.5, 2.5])
    cbar1.ax.set_yticklabels(['Unused', 'Train', 'Test'])

    # 2. 各Foldのクラス分布
    ax2 = fig.add_subplot(gs[1, 0])
    class_ratios = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        y_test = y[test_idx]
        ratios = []
        for class_idx in range(n_classes):
            ratio = np.sum(y_test == class_idx) / len(test_idx) * 100
            ratios.append(ratio)
        class_ratios.append(ratios)

    class_ratios = np.array(class_ratios)
    x = np.arange(n_splits)
    width = 0.25

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for class_idx in range(n_classes):
        ax2.bar(x + class_idx * width, class_ratios[:, class_idx],
                width, label=f'Class {class_idx}', color=colors[class_idx], alpha=0.8)

    # 全体のクラス比率を点線で表示
    for class_idx in range(n_classes):
        overall_ratio = np.sum(y == class_idx) / len(y) * 100
        ax2.axhline(y=overall_ratio, color=colors[class_idx], linestyle='--',
                    linewidth=1.5, alpha=0.5, label=f'Overall Class {class_idx}')

    ax2.set_xlabel('Fold Number', fontsize=12)
    ax2.set_ylabel('Class Ratio (%)', fontsize=12)
    ax2.set_title('Class Distribution in Each Fold (Test Set)', fontsize=13)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(axis='y', alpha=0.3)

    # 3. 各Foldのグループ数とサンプル数
    ax3 = fig.add_subplot(gs[1, 1])
    group_counts = []
    sample_counts = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        groups_test = groups[test_idx]
        group_counts.append(len(np.unique(groups_test)))
        sample_counts.append(len(test_idx))

    x = np.arange(n_splits)
    ax3_twin = ax3.twinx()

    bars1 = ax3.bar(x - 0.2, group_counts, 0.4, label='# Groups', color='#95E1D3', alpha=0.8)
    bars2 = ax3_twin.bar(x + 0.2, sample_counts, 0.4, label='# Samples', color='#F38181', alpha=0.8)

    ax3.set_xlabel('Fold Number', fontsize=12)
    ax3.set_ylabel('Number of Groups', fontsize=12, color='#95E1D3')
    ax3_twin.set_ylabel('Number of Samples', fontsize=12, color='#F38181')
    ax3.set_title('Groups and Samples per Fold (Test Set)', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax3.tick_params(axis='y', labelcolor='#95E1D3')
    ax3_twin.tick_params(axis='y', labelcolor='#F38181')
    ax3.grid(axis='y', alpha=0.3)

    # 凡例を結合
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # 4. グループとクラスのマッピング
    ax4 = fig.add_subplot(gs[2, :])

    # 各グループのクラスを色で表示
    unique_groups = np.unique(groups)
    group_class_map = np.zeros(len(unique_groups))

    for i, group_id in enumerate(unique_groups):
        group_mask = groups == group_id
        # グループ内のクラスを取得（単一クラスのはず）
        group_classes = y[group_mask]
        group_class_map[i] = group_classes[0]

    # グループごとにfold割り当てを表示
    group_fold_assignment = np.full(len(unique_groups), -1)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        groups_test = groups[test_idx]
        for group_id in np.unique(groups_test):
            group_idx = np.where(unique_groups == group_id)[0][0]
            group_fold_assignment[group_idx] = fold

    # 散布図: x=グループID, y=fold, 色=クラス
    scatter = ax4.scatter(unique_groups, group_fold_assignment, c=group_class_map,
                          cmap='viridis', s=150, alpha=0.7, edgecolors='black', linewidth=1)

    ax4.set_xlabel('Group ID', fontsize=12)
    ax4.set_ylabel('Assigned to Test Fold', fontsize=12)
    ax4.set_title('Group Assignment to Folds (colored by class)', fontsize=13)
    ax4.set_yticks(range(n_splits))
    ax4.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax4.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(scatter, ax=ax4, ticks=range(n_classes))
    cbar2.ax.set_yticklabels([f'Class {i}' for i in range(n_classes)])

    plt.suptitle('StratifiedGroupKFold Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('stratified_groupkfold_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'stratified_groupkfold_visualization.png'")


def main():
    """
    メイン処理: StratifiedGroupKFoldの動作確認
    """
    print("=" * 80)
    print("StratifiedGroupKFold の動作確認")
    print("=" * 80)

    # グループ付き不均衡データセットを生成
    print("\nデータセットを生成中...")
    X, y, groups = generate_imbalanced_iris_like_data(
        class_samples=[60, 35, 23],
        random_state=42,
        with_groups=True,
        samples_per_group=5
    )

    print_dataset_info(X, y, "生成されたデータセット", groups=groups)

    # StratifiedGroupKFoldの実行
    n_splits = 5
    print(f"\n{'='*80}")
    print(f"StratifiedGroupKFold (n_splits={n_splits}) の実行")
    print(f"{'='*80}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 各フォールドの分析
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        analyze_fold_distribution(y, groups, train_idx, test_idx, fold)

    # 全体的な統計情報
    print(f"\n{'='*80}")
    print("全体的な統計サマリー")
    print(f"{'='*80}")

    fold_test_sizes = []
    fold_class_ratios = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        y_test = y[test_idx]
        fold_test_sizes.append(len(test_idx))

        # 各クラスの比率を計算
        class_ratios = []
        for class_idx in range(len(np.unique(y))):
            ratio = np.sum(y_test == class_idx) / len(test_idx) * 100
            class_ratios.append(ratio)
        fold_class_ratios.append(class_ratios)

    fold_class_ratios = np.array(fold_class_ratios)

    print(f"\n各foldのテストサンプル数:")
    for fold, size in enumerate(fold_test_sizes, 1):
        print(f"  Fold {fold}: {size}サンプル")
    print(f"  平均: {np.mean(fold_test_sizes):.1f}サンプル")
    print(f"  標準偏差: {np.std(fold_test_sizes):.1f}サンプル")

    print(f"\n各foldのクラス比率（％）:")
    print(f"{'Fold':<8}", end="")
    for class_idx in range(len(np.unique(y))):
        print(f"クラス{class_idx:<8}", end="")
    print()
    print("-" * 80)

    for fold, ratios in enumerate(fold_class_ratios, 1):
        print(f"Fold {fold:<4}", end="")
        for ratio in ratios:
            print(f"{ratio:8.1f}%  ", end="")
        print()

    print(f"\n各クラスの比率の標準偏差（小さいほど均一）:")
    for class_idx in range(len(np.unique(y))):
        std = np.std(fold_class_ratios[:, class_idx])
        print(f"  クラス {class_idx}: {std:.2f}%")

    # 全体のクラス比率と比較
    print(f"\n全体のクラス比率:")
    for class_idx in range(len(np.unique(y))):
        ratio = np.sum(y == class_idx) / len(y) * 100
        print(f"  クラス {class_idx}: {ratio:.1f}%")

    # 可視化
    print(f"\n{'='*80}")
    print("可視化を生成中...")
    print(f"{'='*80}")
    visualize_stratified_groupkfold(y, groups, sgkf, X)


if __name__ == "__main__":
    main()
