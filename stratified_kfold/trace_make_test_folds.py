import pdb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.generate_imbalanced_data import (
    generate_imbalanced_iris_like_data,
    print_dataset_info,
)

# 不均衡なデータを生成
# パターンを変更したい場合は、class_samplesを変更してください
# 例: [100, 30, 10] (極端な不均衡), [53, 37, 27] (5分割しにくい数)
X, y = generate_imbalanced_iris_like_data(class_samples=[53, 37, 27], random_state=42)

print_dataset_info(X, y, "生成されたデータセット")

print("=" * 80)
print("StratifiedKFold の _make_test_folds 処理を追跡")
print("=" * 80)
print()

# パラメータ設定
n_splits = 5
shuffle = False
random_state = 42

print(f"パラメータ:")
print(f"  n_splits = {n_splits}")
print(f"  shuffle = {shuffle}")
print(f"  random_state = {random_state}")
print()

# オリジナルのy
print("=" * 80)
print("ステップ1: オリジナルのy")
print("=" * 80)
print(f"y = {y}")
print(f"yのユニークな値: {np.unique(y)}")
print(f"各クラスのサンプル数: {np.bincount(y)}")
print()

# y_encoded を作成
print("=" * 80)
print("ステップ2: yをエンコード (出現順に0, 1, 2, ...)")
print("=" * 80)
_, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
print(f"y_idx (各ユニーク値の最初の出現インデックス): {y_idx}")
print(f"y_inv (各サンプルがどのユニーク値に対応するか): {y_inv}")

_, class_perm = np.unique(y_idx, return_inverse=True)
print(f"class_perm: {class_perm}")

y_encoded = class_perm[y_inv]
print(f"y_encoded: {y_encoded}")
print(f"  最初の10個: {y_encoded[:10]}")
print(f"  50-60番目: {y_encoded[50:60]}")
print(f"  100-110番目: {y_encoded[100:110]}")
print()

# クラス情報
print("=" * 80)
print("ステップ3: クラス情報の取得")
print("=" * 80)
n_classes = len(y_idx)
y_counts = np.bincount(y_encoded)
print(f"クラス数: {n_classes}")
print(f"各クラスのサンプル数: {y_counts}")
print()

# allocation の計算
print("=" * 80)
print("ステップ4: 各foldへの割り当て (ラウンドロビン方式)")
print("=" * 80)
y_order = np.sort(y_encoded)
print(f"y_order (ソート済み): {y_order}")
print(f"  最初の55個: {y_order[:55]}")
print(f"  95-105番目: {y_order[95:105]}")
print()

print("ラウンドロビン方式で各foldに割り当て:")
allocation = []
for i in range(n_splits):
    fold_samples = y_order[i::n_splits]
    fold_counts = np.bincount(fold_samples, minlength=n_classes)
    allocation.append(fold_counts)
    print(f"  Fold {i}: y_order[{i}::{n_splits}]")
    print(
        f"    取得するインデックス: {list(range(i, len(y_order), n_splits))[:10]}... (最初の10個)"
    )
    print(f"    サンプル: {fold_samples[:10]}... (最初の10個)")
    print(
        f"    各クラスの数: クラス0={fold_counts[0]}, クラス1={fold_counts[1]}, クラス2={fold_counts[2]}"
    )
    print()

allocation = np.asarray(allocation)
print(f"allocation (shape={allocation.shape}):")
print(allocation)
print()

# test_folds の構築
print("=" * 80)
print("ステップ5: 各サンプルにfold番号を割り当て")
print("=" * 80)
test_folds = np.empty(len(y), dtype="i")

for k in range(n_classes):
    print(f"\nクラス {k} の処理:")
    print(f"  allocation[:, {k}] = {allocation[:, k]}")

    # folds_for_classの生成
    folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
    pdb.set_trace()
    print(f"  np.arange({n_splits}) = {np.arange(n_splits)}")
    print(f"  .repeat({list(allocation[:, k])}) =")
    print(f"    {folds_for_class}")
    print(f"    長さ: {len(folds_for_class)} (クラス{k}のサンプル数: {y_counts[k]})")

    # クラスkに属するサンプルのインデックス
    class_k_indices = np.where(y_encoded == k)[0]
    print(f"  クラス{k}のサンプルインデックス: {class_k_indices[:10]}... (最初の10個)")

    # 割り当て
    test_folds[y_encoded == k] = folds_for_class
    print(f"  割り当て完了")

print()
print("=" * 80)
print("ステップ6: 最終的なtest_folds")
print("=" * 80)
print(f"test_folds (各サンプルがどのfoldでテストされるか):")
print(f"  最初の55個: {test_folds[:55]}")
print(f"  50-105番目: {test_folds[50:105]}")
print(f"  100-150番目: {test_folds[100:150]}")
print()

# 各foldの内訳
print("各foldのクラス分布:")
for fold_idx in range(n_splits):
    fold_mask = test_folds == fold_idx
    fold_y = y_encoded[fold_mask]
    fold_counts = np.bincount(fold_y, minlength=n_classes)
    print(
        f"  Fold {fold_idx}: {np.sum(fold_mask)}サンプル "
        f"(クラス0={fold_counts[0]}, クラス1={fold_counts[1]}, クラス2={fold_counts[2]})"
    )
print()

# 実際のStratifiedKFoldと比較
print("=" * 80)
print("検証: 実際のStratifiedKFoldと比較")
print("=" * 80)
skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    test_y = y[test_idx]
    test_counts = np.bincount(test_y, minlength=n_classes)
    print(
        f"Fold {fold_idx}: {len(test_idx)}サンプル "
        f"(クラス0={test_counts[0]}, クラス1={test_counts[1]}, クラス2={test_counts[2]})"
    )

    # test_foldsと一致するか確認
    expected_test_idx = np.where(test_folds == fold_idx)[0]
    if np.array_equal(sorted(test_idx), sorted(expected_test_idx)):
        print(f"  ✓ 手動計算と一致")
    else:
        print(f"  ✗ 手動計算と不一致")

print()
print("=" * 80)
print("完了")
print("=" * 80)
