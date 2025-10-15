"""
不均衡なデータセットを生成するモジュール

Irisデータセット形式（特徴量4次元）で、クラスごとのサンプル数が異なるデータを生成します。
"""

import numpy as np
from typing import Tuple


def generate_imbalanced_iris_like_data(
    class_samples: list[int] = None,
    n_features: int = 4,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iris風の不均衡データセットを生成します。

    Parameters
    ----------
    class_samples : list[int], default=None
        各クラスのサンプル数のリスト。
        Noneの場合は [60, 35, 23] （不均衡なデータ）を使用します。

    n_features : int, default=4
        特徴量の数（Irisに合わせてデフォルトは4）。

    random_state : int, default=None
        乱数シード。

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        特徴量行列。

    y : ndarray of shape (n_samples,)
        クラスラベル。

    Examples
    --------
    >>> X, y = generate_imbalanced_iris_like_data([60, 35, 23], random_state=42)
    >>> X.shape
    (118, 4)
    >>> np.bincount(y)
    array([60, 35, 23])
    """
    if class_samples is None:
        class_samples = [60, 35, 23]

    rng = np.random.RandomState(random_state)
    n_classes = len(class_samples)
    n_samples = sum(class_samples)

    # 各クラスの特徴量を生成（クラスごとに異なる分布）
    X_list = []
    y_list = []

    for class_idx, n_samples_in_class in enumerate(class_samples):
        # クラスごとに中心をずらして正規分布から生成
        center = np.array([class_idx * 2.0] * n_features)
        X_class = rng.randn(n_samples_in_class, n_features) * 0.5 + center
        y_class = np.full(n_samples_in_class, class_idx)

        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # データをシャッフル（オプション）
    # shuffle_idx = rng.permutation(n_samples)
    # X = X[shuffle_idx]
    # y = y[shuffle_idx]

    return X, y


def print_dataset_info(X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
    """
    データセットの情報を表示します。

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        特徴量行列。

    y : ndarray of shape (n_samples,)
        クラスラベル。

    name : str, default="Dataset"
        データセット名。
    """
    print(f"\n{name} 情報:")
    print(f"  サンプル数: {len(X)}")
    print(f"  特徴量数: {X.shape[1]}")
    print(f"  クラス数: {len(np.unique(y))}")
    print(f"  各クラスのサンプル数: {np.bincount(y)}")
    print(f"  クラス比率: {[f'{c:.1%}' for c in np.bincount(y) / len(y)]}")


if __name__ == "__main__":
    # テスト実行
    print("=" * 80)
    print("不均衡データセット生成のテスト")
    print("=" * 80)

    # パターン1: デフォルト（中程度の不均衡）
    X1, y1 = generate_imbalanced_iris_like_data(random_state=42)
    print_dataset_info(X1, y1, "パターン1: デフォルト")

    # パターン2: 極端な不均衡
    X2, y2 = generate_imbalanced_iris_like_data([100, 30, 10], random_state=42)
    print_dataset_info(X2, y2, "パターン2: 極端な不均衡")

    # パターン3: 5分割しにくい数
    X3, y3 = generate_imbalanced_iris_like_data([53, 37, 27], random_state=42)
    print_dataset_info(X3, y3, "パターン3: 5分割しにくい数")

    # パターン4: 4クラス
    X4, y4 = generate_imbalanced_iris_like_data([40, 30, 25, 15], random_state=42)
    print_dataset_info(X4, y4, "パターン4: 4クラス")

    print()
