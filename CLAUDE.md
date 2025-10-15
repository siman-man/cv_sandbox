# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

scikit-learnの交差検証（Cross-Validation）アルゴリズム、特にKFold、StratifiedKFold、GroupKFoldなどの内部実装を理解するための学習用サンドボックスプロジェクト。マルチラベル分類における層化交差検証（`iterative-stratification`ライブラリ）も扱っています。

## 開発環境のセットアップ

このプロジェクトは`uv`を使用して依存関係を管理しています。

```bash
# 依存関係のインストール
uv sync

# Pythonスクリプトの実行
uv run python <script_name>.py
```

## プロジェクト構成

### コアスクリプト

- **iris_kfold.py** - IrisデータセットでKFoldを使った基本的な交差検証の例
- **groupkfold_iris.py** - GroupKFoldの`shuffle=True/False`の動作の違いを比較・可視化
- **multilabel_kfold_example.py** - マルチラベル分類における層化交差検証のデモ
- **trace_make_test_folds.py** - StratifiedKFoldの内部処理(`_make_test_folds`)をステップバイステップでトレース・デバッグするスクリプト

### データ生成ユーティリティ

- **generate_multilabel_data.py** - マルチラベル分類用のサンプルデータを生成
- **generate_imbalanced_data.py** - 不均衡データ（クラス間でサンプル数が異なる）を生成

### 可視化

- **visualize_kfold.py** - KFoldの分割パターンをヒートマップで可視化

### その他

- **translate_docstrings.py** - docstringを翻訳するためのユーティリティ
- **main.py** - 基本的なエントリーポイント（現在はほぼ空）

## アーキテクチャと実装の理解

### scikit-learnの交差検証の仕組み

各交差検証クラスの実装は主に`_iter_test_indices`メソッドまたは`_make_test_folds`メソッドで行われています。READMEに記載されているように：

#### KFold
- データをn分割する単純な実装
- データ数がnで割り切れない場合、余りを先頭から割り振る

#### StratifiedKFold
- クラス比率を維持しながら分割
- 内部処理は`_make_test_folds`で実装
- ラウンドロビン方式で各foldにサンプルを割り当て
- `trace_make_test_folds.py`でステップバイステップの処理を確認可能

#### GroupKFold
- 同じグループのサンプルが異なるfoldに分散しないようにする
- `shuffle=True`: グループをランダムにシャッフルして`np.array_split`で分割（グループ数が等しくなるように分割）
- `shuffle=False`: グループごとのサイズを計算し、重い順に処理。最も軽いビンに追加していく（サンプル個数が等しくなるように分割）

### マルチラベル分類の層化交差検証

- `iterative-stratification`ライブラリの`MultilabelStratifiedKFold`を使用
- 論文: Sechidis et al. (2011) "On the Stratification of Multi-Label Data"
- 各ラベルの割合を維持しながら分割を行う反復層化アルゴリズムを実装
- `multilabel_kfold_example.py`で使用例と各foldのラベル分布の分析を確認可能

## 依存関係

- **scikit-learn** - 機械学習ライブラリ（KFold, StratifiedKFold, GroupKFoldなど）
- **iterative-stratification** - マルチラベル分類用の層化交差検証
- **numpy** - 数値計算
- **matplotlib** - 可視化

## デバッグとトレース

`trace_make_test_folds.py`には`pdb.set_trace()`が含まれており、StratifiedKFoldの内部処理を対話的にデバッグできます。このスクリプトを実行すると、各ステップでの変数の状態を確認できます。

## 日本語コメント

このプロジェクトのコメントとdocstringは日本語で記述されています。新しいコードを追加する際も、日本語でコメントとdocstringを記述してください。
