
## Numpy dot-product and matrix product

在 NumPy 中，点乘和矩阵乘法有不同的表示方式。以下是它们的具体表示方法：

### 1. 点乘（Element-wise Product）

在 NumPy 中，点乘是指逐元素相乘的操作，即两个数组相同位置的元素相乘。

- **表示方法**：使用 `*` 运算符或 `np.multiply()` 函数。
- **适用条件**：两个数组的形状必须相同，或者满足广播（broadcasting）规则。

示例代码：
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 点乘
result = a * b  # 等价于 np.multiply(a, b)
print(result)  # 输出: [ 4 10 18]
```

### 2. 矩阵乘法（Matrix Multiplication）

在 NumPy 中，矩阵乘法是指线性代数中的矩阵相乘操作。

- **表示方法**：使用 `@` 运算符或 `np.dot()` 或 `np.matmul()` 函数。
- **适用条件**：如果 `a` 的形状为 `(m, n)`，则 `b` 的形状应为 `(n, p)`，即前一个矩阵的列数应等于后一个矩阵的行数。

示例代码：
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 矩阵乘法
result = a @ b  # 等价于 np.dot(a, b) 或 np.matmul(a, b)
print(result)
# 输出:
# [[19 22]
#  [43 50]]
```

### 总结

- **点乘**：`*` 或 `np.multiply()`
- **矩阵乘法**：`@`、`np.dot()` 或 `np.matmul()`

这两种运算分别用于逐元素操作和矩阵相乘操作，要注意它们的适用条件和输出结果的不同。


## Numpy padding

在 Python 中，尤其是使用 NumPy 或深度学习框架（如 TensorFlow、PyTorch）时，可以对数组或张量进行填充（padding）操作。以下是几种常见的 padding 操作方法：

### 1. 使用 NumPy 进行 Padding

NumPy 提供了 `np.pad()` 函数，可以用于对数组进行各种形式的填充。

```python
import numpy as np

array = np.array([[1, 2], [3, 4]])

# 对数组进行填充，填充1个元素，使用常量值0
padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=0)
print(padded_array)
# 输出:
# [[0 0 0 0]
#  [0 1 2 0]
#  [0 3 4 0]
#  [0 0 0 0]]
```

- `pad_width`：填充的宽度，可以是整数（表示所有维度相同的填充宽度），也可以是 tuple 或 list 为每一维度指定不同的填充宽度。
- `mode`：填充模式，常用的模式包括 `'constant'`（常量填充）、`'edge'`（边界填充）、`'reflect'`（反射填充）等。
- `constant_values`：当 `mode='constant'` 时，用于指定填充值。

### 2. 在 TensorFlow 中进行 Padding

在 TensorFlow 中，可以使用 `tf.pad()` 进行填充操作。

```python
import tensorflow as tf

tensor = tf.constant([[1, 2], [3, 4]])

# 对 tensor 进行填充，指定填充宽度
padded_tensor = tf.pad(tensor, paddings=[[1, 1], [1, 1]], mode='CONSTANT', constant_values=0)
print(padded_tensor)
# 输出:
# tf.Tensor(
# [[0 0 0 0]
#  [0 1 2 0]
#  [0 3 4 0]
#  [0 0 0 0]], shape=(4, 4), dtype=int32)
```

- `paddings`：一个二维列表，每个子列表指定每个维度的填充宽度，如 `[[1, 1], [1, 1]]` 表示对第一个维度的两侧各填充 1 个元素，对第二个维度的两侧各填充 1 个元素。
- `mode`：填充模式，如 `'CONSTANT'`、`'REFLECT'` 等。
- `constant_values`：当 `mode='CONSTANT'` 时，填充的常量值。

### 3. 在 PyTorch 中进行 Padding

在 PyTorch 中，可以使用 `torch.nn.functional.pad()` 或在 `torch.nn.Conv2d` 等层中直接设置 `padding` 参数。

```python
import torch
import torch.nn.functional as F

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# 对 tensor 进行填充
padded_tensor = F.pad(tensor, pad=(1, 1, 1, 1), mode='constant', value=0)
print(padded_tensor)
# 输出:
# tensor([[0., 0., 0., 0.],
#         [0., 1., 2., 0.],
#         [0., 3., 4., 0.],
#         [0., 0., 0., 0.]])
```

- `pad`：填充宽度的 tuple，格式为 `(left, right, top, bottom)`。
- `mode`：填充模式，如 `'constant'`、`'reflect'`、`'replicate'` 等。
- `value`：常量填充值，当 `mode='constant'` 时指定填充值。

### 4. 自定义 Padding 操作

在某些特定需求下，可以直接通过 NumPy 等方式自定义 padding。例如，手动创建新矩阵并填充边缘。

```python
def custom_padding(array, pad_width, constant_value=0):
    padded_array = np.full((array.shape[0] + 2 * pad_width, array.shape[1] + 2 * pad_width), constant_value)
    padded_array[pad_width:-pad_width, pad_width:-pad_width] = array
    return padded_array

array = np.array([[1, 2], [3, 4]])
padded_array = custom_padding(array, pad_width=1, constant_value=0)
print(padded_array)
```

这种方式可以灵活地定制填充的样式和形状。