import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Irisデータセットの読み込み
iris = load_iris()
X, y = iris.data, iris.target

# KFoldの設定（5分割）
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 各foldの結果を保存
fold_scores = []

print("KFold Cross Validation - 学習と評価")
print("=" * 50)

# KFoldで分割して学習・評価
for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
    # 訓練データとテストデータに分割
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # モデルの学習
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # 予測と評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_scores.append(accuracy)

    print(f"Fold {fold}: Accuracy = {accuracy:.4f}")
    print(f"  訓練サンプル数: {len(train_idx)}, テストサンプル数: {len(test_idx)}")

print("=" * 50)
print(f"平均精度: {np.mean(fold_scores):.4f}")
print(f"標準偏差: {np.std(fold_scores):.4f}")
