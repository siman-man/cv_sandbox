"""
iterative-stratificationを使ったマルチラベル分類のK-Fold交差検証サンプル
"""
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.generate_multilabel_data import generate_multilabel_data, print_label_statistics


def analyze_fold_distribution(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, fold_num: int) -> None:
    """
    各フォールドのラベル分布を分析して表示

    Parameters
    ----------
    y : np.ndarray
        マルチラベル行列
    train_idx : np.ndarray
        訓練データのインデックス
    test_idx : np.ndarray
        テストデータのインデックス
    fold_num : int
        フォールド番号
    """
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_num}")
    print(f"{'='*60}")
    print(f"訓練データ数: {len(train_idx)}, テストデータ数: {len(test_idx)}")

    print(f"\n各ラベルの分布:")
    print(f"{'ラベル':<10} {'訓練':<15} {'テスト':<15} {'全体':<15}")
    print(f"{'-'*60}")

    n_labels = y.shape[1]
    for i in range(n_labels):
        train_count = y_train[:, i].sum()
        test_count = y_test[:, i].sum()
        total_count = y[:, i].sum()

        train_ratio = train_count / len(train_idx) * 100
        test_ratio = test_count / len(test_idx) * 100
        total_ratio = total_count / len(y) * 100

        print(f"ラベル {i:<4} "
              f"{train_count:4d} ({train_ratio:5.1f}%)  "
              f"{test_count:4d} ({test_ratio:5.1f}%)  "
              f"{total_count:4d} ({total_ratio:5.1f}%)")


def main():
    """
    メイン処理: iterative-stratificationを使った交差検証のデモ
    """
    print("マルチラベルデータの生成中...")
    X, y = generate_multilabel_data(
        n_samples=1000,
        n_features=20,
        n_labels=5,
        random_state=42
    )

    print("\n" + "="*60)
    print("データセット全体の統計")
    print("="*60)
    print_label_statistics(y)

    # MultilabelStratifiedKFoldを使った交差検証
    n_splits = 5
    print(f"\n{'='*60}")
    print(f"MultilabelStratifiedKFold (n_splits={n_splits}) の実行")
    print(f"{'='*60}")

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 各フォールドの分析
    for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y), 1):
        analyze_fold_distribution(y, train_idx, test_idx, fold)

    # ラベルの組み合わせパターンの分析
    print(f"\n{'='*60}")
    print("ラベル組み合わせパターンの分析")
    print(f"{'='*60}")

    # ラベルの組み合わせをタプルに変換
    label_combinations = [tuple(row) for row in y]
    unique_combinations, counts = np.unique(label_combinations, axis=0, return_counts=True)

    print(f"ユニークなラベル組み合わせ数: {len(unique_combinations)}")
    print(f"\n最も頻出する上位5つの組み合わせ:")
    sorted_idx = np.argsort(counts)[::-1][:5]

    for idx in sorted_idx:
        combination = unique_combinations[idx]
        count = counts[idx]
        labels = [i for i, val in enumerate(combination) if val == 1]
        print(f"  ラベル {labels}: {count}回 ({count/len(y)*100:.1f}%)")


if __name__ == "__main__":
    main()
