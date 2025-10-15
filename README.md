## 全体の処理

色々抽象化されているが、大体は各フォールド分けのクラスの `_iter_test_indices` で実装が行われている。なので個々の処理の確認することで大体分かる。

## KFold

データを n 分割するメソッド。特に難しいことはしておらず単純に N 分割するだけ。

データ数が n で分割出来ないときはあまりを先頭から割り振っていく。

## GroupKFold

フォールド分けする際には同じ別々のフォールドに分けられると困るケースがある。

同一人物から得られたデータ等でその人の特性を学習したうえで、その人のデータを推測してしまうと汎化性能を確認出来なくなってしまう。

GroupKFold は shuffle の指定あり、なしで挙動が変化する

### shuffle あり

グループをランダムにシャッフルして np.array_split で分割処理を行う。

### shuffle なし

グループ毎にサイズを求めそれを重い順から処理していく。n 個のビンが存在する場合に一番軽いビンに対して追加を行う。

shuffle の True との違いは、True の場合は「グループの数」が等しくなるように分割され、False の場合は「サンプルの個数」が等しくなるように分割される。

## StratifiedKFold

各クラスをラウンドロビン方式でフォールド毎に割り振って、それを各クラスごとで結合する。

## StratifiedGroupKFold

各クラスの比率を保ったままグループの制約も追加したもの。

比率というがクラスごとにフォールド間のサンプルの標準偏差を小さくする処理を行っているだけではある。

### _find_best_fold

以下の基準で最良のfoldを選択：
  - 第1優先: fold_evalが最小（クラス比率が最も均等になる）
  - 第2優先: 評価が同じ場合、サンプル数が少ないfoldを選ぶ（fold間のサンプル数バランスも考慮）

```python
    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # 提案された各foldでのクラス間の分布を要約
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = fold_eval < min_eval or (
                np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold
```


## TimeSeriesSplit

説明するか不明


### 同一サイズの確認処理

check_consistent_length

https://github.com/scikit-learn/scikit-learn/blob/5bbbc77f97a8b362e12e4cc365ae35780cfcc3f7/sklearn/utils/validation.py#L445

内部的には各配列を受け取ってそれらのサイズを計算したあとに、集合を取ってそのサイズが 1 かどうかを確認している
