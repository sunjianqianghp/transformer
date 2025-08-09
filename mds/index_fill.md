`torch.Tensor.index_fill_()` 是 PyTorch 中一个用于 **原地操作（in-place）** 的方法，它可以沿指定的维度，将张量中某些位置的值替换为给定的值。

---

### 🔧 语法：

```python
tensor.index_fill_(dim, index, value)
```

---

### ✅ 参数说明：

* `dim`：`int`，指定在哪个维度上进行操作。
* `index`：`1D Tensor`，包含了要修改位置的索引。
* `value`：`float` 或 `Tensor`，要填充的值。

> ⚠️ 注意：这个方法是原地操作，调用的是 `index_fill_()`（后缀有下划线）。也有非原地版本：`index_fill()`。

---

### 🧠 功能描述：

对张量 `tensor` 沿着维度 `dim`，在 `index` 所指定的位置上，填入 `value`。

---

### 📌 示例：

```python
import torch

x = torch.zeros(3, 5)
index = torch.tensor([1, 3])
x.index_fill_(1, index, 7)
```

#### 输出：

```python
tensor([[0., 7., 0., 7., 0.],
        [0., 7., 0., 7., 0.],
        [0., 7., 0., 7., 0.]])
```

#### 解释：

* `x` 是一个形状为 `[3, 5]` 的 0 矩阵。
* 沿着 `dim=1`（即列方向），在第 `1` 和第 `3` 列，用 `7` 替换原值。
* 所以每一行的第 `1` 和第 `3` 列都被改为 `7`。

---

### ✅ 非原地版本：

如果你不想修改原 tensor，可以使用非原地版本：

```python
x2 = x.index_fill(1, index, 7)
```

---

需要我帮你举一个更复杂的例子或配合其他操作使用吗？
