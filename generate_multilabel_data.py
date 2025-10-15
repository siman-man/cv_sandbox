"""
マルチラベル分類用のサンプルデータを生成するモジュール
"""
import numpy as np
from typing import Tuple


def generate_multilabel_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_labels: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチラベル分類用のサンプルデータを生成

    Parameters
    ----------
    n_samples : int
        サンプル数
    n_features : int
        特徴量の数
    n_labels : int
        ラベルの数
    random_state : int
        乱数シード

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        特徴量行列
    y : np.ndarray, shape (n_samples, n_labels)
        マルチラベル行列（各要素は0または1）
    """
    rng = np.random.RandomState(random_state)

    # 特徴量を生成
    X = rng.randn(n_samples, n_features)

    # マルチラベルを生成
    # 各サンプルに対して1〜3個のラベルをランダムに割り当て
    y = np.zeros((n_samples, n_labels), dtype=int)

    for i in range(n_samples):
        # 各サンプルに割り当てるラベル数をランダムに決定（1〜3個）
        n_labels_for_sample = rng.randint(1, min(4, n_labels + 1))
        # ランダムにラベルを選択
        selected_labels = rng.choice(n_labels, size=n_labels_for_sample, replace=False)
        y[i, selected_labels] = 1

    return X, y


def print_label_statistics(y: np.ndarray) -> None:
    """
    ラベルの統計情報を表示

    Parameters
    ----------
    y : np.ndarray
        マルチラベル行列
    """
    n_samples, n_labels = y.shape

    print(f"サンプル数: {n_samples}")
    print(f"ラベル数: {n_labels}")
    print(f"\n各ラベルの出現数:")
    for i in range(n_labels):
        count = y[:, i].sum()
        print(f"  ラベル {i}: {count} ({count/n_samples*100:.1f}%)")

    print(f"\n各サンプルのラベル数の分布:")
    labels_per_sample = y.sum(axis=1)
    for i in range(1, labels_per_sample.max() + 1):
        count = (labels_per_sample == i).sum()
        print(f"  {i}個: {count} ({count/n_samples*100:.1f}%)")


if __name__ == "__main__":
    # テスト用のコード
    X, y = generate_multilabel_data(n_samples=100, n_labels=5)
    print_label_statistics(y)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\n最初の5サンプルのラベル:")
    print(y[:5])
