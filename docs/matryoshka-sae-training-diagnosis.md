# Matryoshka SAE 架构、初始化与 Dense Feature 问题分析

## 实际架构

`MatryoshkaSparseAutoEncoder` 继承自普通 `SparseAutoEncoder`：

```python
class MatryoshkaSparseAutoEncoder(SparseAutoEncoder):
    ...
```

因此，encoder、decoder、TopK 和参数初始化都来自普通 SAE。

每个 batch 只在完整的 32768 维 latent 上进行一次全局 TopK：

```python
feature_acts = TopK(hidden_pre[..., :32768], k=128)
```

不同宽度的 Matryoshka reconstruction 只是截取同一个 `feature_acts` 的前缀：

```python
feature_acts[..., :width] @ W_D[:width]
```

相关实现见 `src/llamascopium/models/matryoshka_sae.py` 中的 `decode_prefix()` 和 `compute_loss()`。

所以，当前实现不是每个 prefix 分别执行 TopK，而是：

```text
全局 TopK 一次
→ 截取不同 prefix
→ 计算多个 reconstruction loss
```

这个设计会强烈鼓励前缀 feature 抢占全局 TopK 槽位。

## 当前完整初始化流程

### 1. 创建普通 32768-width SAE 参数

Matryoshka SAE 没有覆盖 `init_parameters()`，因此使用普通 SAE 的初始化：

```python
W_E ~ Uniform(-1 / sqrt(32768), 1 / sqrt(32768))
W_D ~ Uniform(-1 / sqrt(2048), 1 / sqrt(2048))
b_E = 0
b_D = 0
```

### 2. Encoder 和 decoder 独立随机初始化

Workflow 当前设置为：

```python
init_encoder_with_decoder_transpose = False
```

因此：

```python
W_E != W_D.T
```

Encoder 选中的随机 feature 与它对应的 decoder direction 在初始时没有对齐。在严格全局 TopK 下，这会让早期的随机优势更容易固化。

### 3. Dataset-wise activation normalization

框架首先估计 Qwen layer 27 residual activation 的平均 L2 norm，然后把训练输入缩放到平均 norm：

$$
\sqrt{d_{\text{model}}} = \sqrt{2048} \approx 45.25
$$

后续的 decoder bias 初始化和 decoder norm grid search 都使用归一化后的 activation。

### 4. Decoder bias 初始化

Matryoshka SAE 继承了 `SparseAutoEncoder`，因此可以通过：

```python
isinstance(sae, SparseAutoEncoder)
```

然后执行：

```python
b_D = label.mean(...)
```

需要注意，虽然 workflow 中的配置名是：

```python
bias_init_method = "geometric_median"
```

但当前实现实际计算的是普通 mean，而不是 geometric median。

### 5. 搜索统一 decoder norm

由于：

```python
grid_search_init_norm = True
```

初始化器会先粗搜索：

```text
0.1, 0.2, ..., 5.0
```

然后在最佳值附近以约 `0.01` 的间隔细搜索。

每个候选值都会把所有 decoder feature 设为相同 norm：

```python
W_D[i] *= candidate_norm / ||W_D[i]||
```

但当前初始化器使用的目标是：

```python
sae.compute_loss(batch)["l_rec"].mean()
```

在 Matryoshka `compute_loss()` 中：

- `l_rec` 仍然是完整 32768-width reconstruction loss；
- 内层 prefix loss 保存在 `l_matryoshka`；
- 最终的加权总目标保存在 `loss`。

因此，当前 grid search 只根据完整 32768-width SAE 的 reconstruction loss 选择 decoder norm，完全忽略了：

```text
width 2048 loss
width 4096 loss
width 8192 loss
width 16384 loss
```

这是普通 SAE 的 norm initialization，而不是 Matryoshka-aware initialization。

### 6. Encoder bias 初始化

最后执行：

```python
b_E -= hidden_pre.mean(dim=0)
```

目标是使每个 feature 的平均 pre-activation 接近 0。

但它使用完整 32768-width encoder 的全局路径，只对齐了 pre-activation mean，没有对齐：

- 每个 feature 的 variance；
- encoder norm；
- 不同 prefix 获得的 TopK 槽位数；
- 不同 prefix 之间的梯度大小。

## Matryoshka loss 权重

Workflow 没有显式设置：

```python
matryoshka_loss_weights = None
```

配置会生成等权并归一化的权重：

```python
[0.2, 0.2, 0.2, 0.2, 0.2]
```

它们分别对应：

```text
2048
4096
8192
16384
32768
```

总 reconstruction loss 为：

$$
L = 0.2L_{2048} + 0.2L_{4096} + 0.2L_{8192} + 0.2L_{16384} + 0.2L_{32768}
$$

因为权重已归一化，总 loss 尺度没有简单放大五倍。

但是，不同位置的 feature 参与 reconstruction loss 的次数仍不均衡：

| Feature 区间 | 参与的 loss 数量 |
|---|---:|
| `0:2048` | 5 |
| `2048:4096` | 4 |
| `4096:8192` | 3 |
| `8192:16384` | 2 |
| `16384:32768` | 1 |

## 为什么会产生 Prefix TopK 抢占

随机初始化时，全局 TopK 的 128 个槽位近似均匀分布在 32768 个 feature 中。

因此，每个 prefix 在初始时获得的 active feature 数量期望为：

| Prefix width | 初始 active feature 数量期望 |
|---:|---:|
| 2048 | 8 |
| 4096 | 16 |
| 8192 | 32 |
| 16384 | 64 |
| 32768 | 128 |

`width=2048` 的 loss 权重与完整 SAE 一样是 `0.2`，但它初始时只能使用约 8 个 active feature 重构 2048 维 residual activation。

8 个随机且 encoder/decoder 没有对齐的 feature 很难做好重构。因此，$L_{2048}$ 会产生强烈的优化压力，使前 2048 个 feature 更容易进入全局 TopK。

这会形成正反馈：

```text
前缀 feature 更容易进入 TopK
→ prefix reconstruction 改善
→ 这些 feature 同时得到多个 prefix loss 的梯度
→ pre-activation 进一步增大
→ 更稳定地占据全局 TopK
```

最终，前 2048 维中的一批 feature 可能永久占据 128 个全局 TopK 槽位中的大部分。这与观察到的约 100 个 everywhere-active dense feature 高度一致。

这些 dense feature 不是因为最小 prefix 单独执行了 `TopK=128`，而是因为最小 prefix 只有在大量前缀 feature 抢到全局 TopK 时，才能显著降低它的 reconstruction loss。

## 初始化中的两个具体缺陷

### 缺陷一：Grid search 忽略 Matryoshka reconstruction loss

当前实现：

```python
losses[norm] = sae.compute_loss(batch)["l_rec"].mean()
```

更符合 Matryoshka 设定的做法是使用纯 reconstruction 的加权总目标：

```python
ctx = sae.compute_loss(batch, auxk_coefficient=0.0)
losses[norm] = (
    ctx["l_matryoshka"]
    + sae.full_matryoshka_loss_weight * ctx["l_rec"]
).mean()
```

不建议无条件直接使用 `ctx["loss"]`，因为未来它可能同时包含 sparsity regularization 或其他辅助目标。

### 缺陷二：Encoder 和 decoder 随机且不对齐

当前组合是：

```python
init_encoder_with_decoder_transpose = False
initial_k = 128
top_k = 128
```

这表示没有 encoder-decoder direction alignment，也没有 k annealing。随机 feature 从第一步开始就在严格的 128 个槽位中竞争，早期赢家很容易固化。

可以对照测试：

```python
init_encoder_with_decoder_transpose = True
initial_k = 512
top_k = 128
```

这能让更多 feature 在训练早期获得梯度，再逐步退火到 `k=128`。

## 更根本的问题

即使改善初始化，当前目标仍然要求仅依靠全局 TopK 中落在前 2048 维的 feature 完成 $L_{2048}$，而 $L_{2048}$ 与 $L_{32768}$ 的权重相同。

因此，更直接的改动是降低小 prefix 的权重，例如：

```python
matryoshka_loss_weights = [
    0.025,  # 2048
    0.05,   # 4096
    0.10,   # 8192
    0.20,   # 16384
    0.625,  # 32768
]
```

另一种方案是修改架构，让每个 prefix 使用与宽度成比例的独立 TopK：

| Prefix width | TopK |
|---:|---:|
| 2048 | 8 |
| 4096 | 16 |
| 8192 | 32 |
| 16384 | 64 |
| 32768 | 128 |

这样可以让每个宽度的平均 activation density 都约为：

$$
\frac{128}{32768} \approx 0.39\%
$$

## 结论

当前 dense collapse 不只是初始化噪声，而是以下组合所带来的直接优化结果：

```text
全局 TopK 后截取 prefix
+ 小 prefix 与完整 SAE 等权
+ 前缀 feature 参与更多 reconstruction loss
+ 随机且不对齐的 encoder/decoder 初始化
+ 从第一步开始严格使用 TopK=128
```

初始化方式会加速 dense feature 的形成，但根本压力来自当前 Matryoshka prefix loss 与全局 TopK 的组合。
