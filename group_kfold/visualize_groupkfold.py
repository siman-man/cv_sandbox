"""
GroupKFoldのフォールド分けをビジュアライズするプログラム
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from pathlib import Path


def create_group_data(group_sizes):
    """
    グループベースのデータを作成します。

    Parameters
    ----------
    group_sizes : list[int]
        各グループのサンプル数。

    Returns
    -------
    X : ndarray
        特徴量（ダミー）。
    y : ndarray
        ラベル（ダミー）。
    groups : ndarray
        グループID。
    """
    n_samples = sum(group_sizes)
    X = np.zeros((n_samples, 4))  # ダミーの特徴量
    y = np.zeros(n_samples)  # ダミーのラベル
    groups = np.zeros(n_samples, dtype=int)

    current_idx = 0
    for group_id, size in enumerate(group_sizes):
        groups[current_idx : current_idx + size] = group_id
        current_idx += size

    return X, y, groups


def visualize_groupkfold(
    X,
    y,
    groups,
    n_splits=5,
    shuffle_values=[False, True],
    random_state=42,
    save_dir="group_kfold/images",
):
    """
    GroupKFoldのフォールド分けをビジュアライズします。

    Parameters
    ----------
    X : ndarray
        特徴量。
    y : ndarray
        ラベル。
    groups : ndarray
        グループID。
    n_splits : int, default=5
        分割数。
    shuffle_values : list[bool], default=[False, True]
        比較するshuffleの値。
    random_state : int, default=42
        乱数シード。
    save_dir : str, default="images"
        保存先ディレクトリ。
    """
    n_samples = len(X)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # 各shuffleパターンを個別の画像として保存
    for shuffle in shuffle_values:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        # GroupKFoldの作成
        if shuffle:
            gkf = GroupKFold(n_splits=n_splits)
            # シャッフルの実装のため、random_stateを使用
            np.random.seed(random_state)
        else:
            gkf = GroupKFold(n_splits=n_splits)

        # フォールド分けデータの作成
        fold_data = np.zeros((n_splits, n_samples))

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            fold_data[fold, train_idx] = 1  # 訓練データ
            fold_data[fold, test_idx] = 2  # テストデータ

        # カラーマップ（訓練=1、テスト=2）
        cmap = plt.cm.colors.ListedColormap(["skyblue", "coral"])
        bounds = [0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # ヒートマップ
        im = ax.imshow(
            fold_data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
        )

        # グループの境界線とラベルを描画
        unique_groups = np.unique(groups)
        for group_id in unique_groups:
            group_indices = np.where(groups == group_id)[0]
            if len(group_indices) > 0:
                # グループの開始位置に境界線
                start_idx = group_indices[0]
                end_idx = group_indices[-1]

                # 境界線（太い黒線）
                if start_idx > 0:
                    ax.axvline(x=start_idx - 0.5, color="black", linewidth=2, linestyle="-")
                if end_idx + 1 < n_samples:
                    ax.axvline(x=end_idx + 0.5, color="black", linewidth=2, linestyle="-")

                # グループIDを表示（上部）
                center_x = (start_idx + end_idx) / 2
                ax.text(
                    center_x, -0.7, f"G{group_id}",
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
                )

                # グループの背景色を交互に変える（薄いグレー）
                if group_id % 2 == 0:
                    ax.axvspan(start_idx - 0.5, end_idx + 0.5,
                             facecolor="gray", alpha=0.1, zorder=0)

        # 軸の設定
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Fold Number", fontsize=12)
        title = f"GroupKFold (n_splits={n_splits}, shuffle={shuffle})"
        ax.set_title(title, fontsize=14, pad=30)

        ax.set_yticks(range(n_splits))
        ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])

        # Y軸の範囲を調整（グループラベルのための余白を確保）
        ax.set_ylim(n_splits - 0.5, -1.5)
        ax.set_xlim(-0.5, n_samples - 0.5)

        # グリッド
        ax.set_xticks(np.arange(0, n_samples, 10), minor=True)
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

        # 統計情報を表示
        stats_text = "Sample counts per fold:\n"
        for fold in range(n_splits):
            n_test = np.sum(fold_data[fold] == 2)
            test_groups = np.unique(groups[fold_data[fold] == 2])
            stats_text += f"  Fold {fold+1}: {n_test} samples, groups {list(test_groups)}\n"

        ax.text(
            1.01,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # カラーバー
        cbar = plt.colorbar(im, ax=ax, ticks=[1, 2], pad=0.01, fraction=0.03)
        cbar.ax.set_yticklabels(["Train", "Test"])

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # 保存（shuffleの値をファイル名に含める）
        shuffle_str = "shuffle" if shuffle else "noshuffle"
        filename = f"groupkfold_n{n_splits}_{shuffle_str}.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()


def print_groupkfold_statistics(X, y, groups, n_splits=5):
    """
    GroupKFoldの統計情報を表示します。

    Parameters
    ----------
    X : ndarray
        特徴量。
    y : ndarray
        ラベル。
    groups : ndarray
        グループID。
    n_splits : int, default=5
        分割数。
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    print("=" * 80)
    print("データセット情報")
    print("=" * 80)
    print(f"総サンプル数: {len(X)}")
    print(f"グループ数: {n_groups}")
    print(f"各グループのサンプル数:")
    for group_id in unique_groups:
        n_samples_in_group = np.sum(groups == group_id)
        print(f"  Group {group_id}: {n_samples_in_group} samples")
    print()

    # shuffle=False
    print("=" * 80)
    print(f"GroupKFold (n_splits={n_splits}, shuffle=False)")
    print("=" * 80)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        test_groups = np.unique(groups[test_idx])
        train_groups = np.unique(groups[train_idx])
        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
        print(f"    Groups: {list(train_groups)}")
        print(f"  Test:  {len(test_idx)} samples, {len(test_groups)} groups")
        print(f"    Groups: {list(test_groups)}")
        print(f"    Group sizes: {[np.sum(groups == g) for g in test_groups]}")


if __name__ == "__main__":
    print("=" * 80)
    print("GroupKFold ビジュアライゼーション")
    print("=" * 80)
    print()

    # パターン1: 均等なグループサイズ
    print("\n【パターン1: 均等なグループサイズ】")
    group_sizes_1 = [10] * 10  # 10個のグループ、各10サンプル
    X1, y1, groups1 = create_group_data(group_sizes_1)
    print_groupkfold_statistics(X1, y1, groups1, n_splits=5)
    visualize_groupkfold(X1, y1, groups1, n_splits=5, shuffle_values=[False])

    # パターン2: 不均等なグループサイズ
    print("\n【パターン2: 不均等なグループサイズ】")
    group_sizes_2 = [5, 5, 8, 10, 12, 15, 20, 8, 7, 10]  # 10個のグループ、異なるサイズ
    X2, y2, groups2 = create_group_data(group_sizes_2)
    print_groupkfold_statistics(X2, y2, groups2, n_splits=5)
    visualize_groupkfold(X2, y2, groups2, n_splits=5, shuffle_values=[False])

    # パターン3: shuffle=TrueとFalseの比較
    print("\n【パターン3: shuffle=True vs False の比較】")
    group_sizes_3 = [8, 12, 6, 15, 10, 9, 14, 7, 11, 8]
    X3, y3, groups3 = create_group_data(group_sizes_3)
    visualize_groupkfold(
        X3, y3, groups3, n_splits=5, shuffle_values=[False, True], random_state=42
    )

    print("\n全てのビジュアライゼーションを images/ ディレクトリに保存しました。")
