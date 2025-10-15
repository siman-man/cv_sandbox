## KFold

X個のサンプルを n分割するだけ。

`y`, `groups` は互換性のためだけに存在しており使用されていない。

```python
    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop
```

`full` を使用して均等に割り当てを行う。

```python
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
```

割り切れない場合はあまりの部分を先頭から割り当てる

```python
        fold_sizes[: n_samples % n_splits] += 1
```
