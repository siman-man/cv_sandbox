"""
不均衡なデータセットを生成するモジュール

Irisデータセット形式（特徴量4次元）で、クラスごとのサンプル数が異なるデータを生成します。
"""

import numpy as np
from typing import Tuple, overload


@overload
def generate_imbalanced_iris_like_data(
    class_samples: list[int] = None,
    n_features: int = 4,
    random_state: int = None,
    with_groups: bool = False,
    samples_per_group: int = 5
) -> Tuple[np.ndarray, np.ndarray]: ...

@overload
def generate_imbalanced_iris_like_data(
    class_samples: list[int] = None,
    n_features: int = 4,
    random_state: int = None,
    *,
    with_groups: bool = True,
    samples_per_group: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def generate_imbalanced_iris_like_data(
    class_samples: list[int] = None,
    n_features: int = 4,
    random_state: int = None,
    with_groups: bool = False,
    samples_per_group: int = 5
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    with_groups : bool, default=False
        Trueの場合、グループ情報も返します。

    samples_per_group : int, default=5
        各グループに含まれるサンプル数の目安。
        実際のグループサイズは、クラス内でサンプルを均等に分割するため多少異なる場合があります。

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        特徴量行列。

    y : ndarray of shape (n_samples,)
        クラスラベル。

    groups : ndarray of shape (n_samples,), optional
        グループラベル（with_groups=Trueの場合のみ返される）。
        各クラス内で複数のグループに分割されます。

    Examples
    --------
    >>> X, y = generate_imbalanced_iris_like_data([60, 35, 23], random_state=42)
    >>> X.shape
    (118, 4)
    >>> np.bincount(y)
    array([60, 35, 23])

    >>> X, y, groups = generate_imbalanced_iris_like_data([60, 35, 23], random_state=42, with_groups=True)
    >>> len(np.unique(groups))
    24
    """
    if class_samples is None:
        class_samples = [60, 35, 23]

    rng = np.random.RandomState(random_state)
    n_classes = len(class_samples)
    n_samples = sum(class_samples)

    # 各クラスの特徴量を生成（クラスごとに異なる分布）
    X_list = []
    y_list = []
    groups_list = []
    group_id = 0

    for class_idx, n_samples_in_class in enumerate(class_samples):
        # クラスごとに中心をずらして正規分布から生成
        center = np.array([class_idx * 2.0] * n_features)
        X_class = rng.randn(n_samples_in_class, n_features) * 0.5 + center
        y_class = np.full(n_samples_in_class, class_idx)

        X_list.append(X_class)
        y_list.append(y_class)

        if with_groups:
            # クラス内でグループを作成
            n_groups_in_class = max(1, n_samples_in_class // samples_per_group)
            # 各サンプルをグループに割り当て
            class_groups = np.array_split(np.arange(n_samples_in_class), n_groups_in_class)
            groups_class = np.zeros(n_samples_in_class, dtype=int)
            for i, group_indices in enumerate(class_groups):
                groups_class[group_indices] = group_id
                group_id += 1
            groups_list.append(groups_class)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # データをシャッフル（オプション）
    # shuffle_idx = rng.permutation(n_samples)
    # X = X[shuffle_idx]
    # y = y[shuffle_idx]

    if with_groups:
        groups = np.hstack(groups_list)
        return X, y, groups
    else:
        return X, y


def print_dataset_info(X: np.ndarray, y: np.ndarray, name: str = "Dataset", groups: np.ndarray | None = None):
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

    groups : ndarray of shape (n_samples,), optional
        グループラベル。
    """
    print(f"\n{name} 情報:")
    print(f"  サンプル数: {len(X)}")
    print(f"  特徴量数: {X.shape[1]}")
    print(f"  クラス数: {len(np.unique(y))}")
    print(f"  各クラスのサンプル数: {np.bincount(y)}")
    print(f"  クラス比率: {[f'{c:.1%}' for c in np.bincount(y) / len(y)]}")

    if groups is not None:
        print(f"  グループ数: {len(np.unique(groups))}")
        print(f"  各グループのサンプル数:")
        for group_id in np.unique(groups):
            group_mask = groups == group_id
            group_size = np.sum(group_mask)
            group_classes = y[group_mask]
            class_dist = np.bincount(group_classes, minlength=len(np.unique(y)))
            print(f"    グループ {group_id}: {group_size}サンプル (クラス分布: {class_dist})")


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

    # パターン5: グループ付き
    print("\n" + "=" * 80)
    print("グループ付きデータセット生成のテスト")
    print("=" * 80)
    X5, y5, groups5 = generate_imbalanced_iris_like_data(
        [60, 35, 23], random_state=42, with_groups=True, samples_per_group=5
    )
    print_dataset_info(X5, y5, "パターン5: グループ付き", groups=groups5)

    print()
