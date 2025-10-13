"""
_split.pyの全てのdocstringを日本語に翻訳するスクリプト
"""
import re

# 翻訳辞書
translations = {
    # 一般的なフレーズ
    "Always ignored, exists for compatibility.": "常に無視されます。互換性のために存在します。",
    "Returns the number of splitting iterations in the cross-validator.": "交差検証器の分割イテレーション回数を返します。",
    "Training data, where `n_samples` is the number of samples\n            and `n_features` is the number of features.": "訓練データ。`n_samples` はサンプル数、\n            `n_features` は特徴量の数です。",
    "The target variable for supervised learning problems.": "教師あり学習問題の目的変数。",
    "Group labels for the samples used while splitting the dataset into\n            train/test set.": "データセットを訓練/テストセットに分割する際に使用される\n            サンプルのグループラベル。",
    "The training set indices for that split.": "その分割の訓練セットのインデックス。",
    "The testing set indices for that split.": "その分割のテストセットのインデックス。",

    # クラスdocstring
    "Leave P Group(s) Out cross-validator.": "P個のグループを除外する交差検証器。",
    "Repeated K-Fold cross validator.": "繰り返しK-分割交差検証器。",
    "Repeated Stratified K-Fold cross validator.": "繰り返し層化K-分割交差検証器。",
    "Shuffle-Group(s)-Out cross-validation iterator.": "シャッフルグループ除外交差検証イテレータ。",
    "Stratified ShuffleSplit cross-validator.": "層化シャッフル分割交差検証器。",
    "Predefined split cross-validator.": "事前定義された分割交差検証器。",

    # パラメータの説明
    "Number of folds. Must be at least 2.": "分割数。最低2以上である必要があります。",
    "Whether to shuffle the data before splitting into batches.": "バッチに分割する前にデータをシャッフルするかどうか。",
    "Number of re-shuffling & splitting iterations.": "再シャッフルと分割のイテレーション回数。",
    "The number of samples to include in the test set.": "テストセットに含めるサンプル数。",
    "The number of samples to include in the train set.": "訓練セットに含めるサンプル数。",
}

def translate_file(filepath):
    """ファイルを読み込んで翻訳を適用"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 翻訳を適用
    for english, japanese in translations.items():
        content = content.replace(english, japanese)

    # ファイルに書き戻す
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"翻訳完了: {filepath}")

if __name__ == "__main__":
    filepath = "/Users/siman/Programming/python/cv_sandbox/.venv/lib/python3.12/site-packages/sklearn/model_selection/_split.py"
    translate_file(filepath)
