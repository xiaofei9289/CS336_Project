目前学习的 Llama 风格、Decoder-only Transformer 中，一个 `TransformerBlock` 主要包含两大模块：

1. 多头因果自注意力 `Multi-Head Self-Attention`
2. 前馈神经网络 `SwiGLU`

同时配合：

- 两个 `RMSNorm`
- 两次残差连接
- `RoPE` 位置编码
- 因果掩码

**一个 Llama 风格的 TransformerBlock 由“多头因果自注意力”和“SwiGLU 前馈网络”两个主要子模块组成。两个子模块之前分别使用 RMSNorm，之后分别通过残差连接与原输入相加。Attention 负责让不同 token 之间交换和聚合信息，SwiGLU 负责进一步加工每个 token 自身的特征。RoPE 为 Q、K 加入位置信息，因果掩码保证每个位置只能查看自己及之前的 token。整个 Block 不改变输入形状，输入和输出通常都是 `(batch_size, seq_len, d_model)`。**

```
Token IDs                         (batch, seq)
   ↓
Embedding
   ↓
初始隐藏向量 x₀                  (batch, seq, d_model)
   ↓
Transformer Block × num_layers
   │
   │  Attention 分支
   ├── RMSNorm(x)
   ├── Q、K、V 线性投影
   ├── 将 Q、K、V 拆分成多个注意力头
   ├── 对 Q、K 应用 RoPE
   ├── QKᵀ / √dₖ
   ├── Causal Mask
   ├── Softmax
   ├── Attention 权重与 V 加权求和
   ├── 合并多个注意力头
   ├── 输出投影
   └── 与分支输入进行残差连接
          ↓
       x₁ = x + Attention(RMSNorm(x))
          │
          │  FFN 分支
          ├── RMSNorm(x₁)
          ├── SwiGLU
          └── 与分支输入进行残差连接
                 ↓
              x₂ = x₁ + SwiGLU(RMSNorm(x₁))
   ↓
下一个 Transformer Block
   ↓
最终隐藏向量
   ↓
最终 RMSNorm
   ↓
LM Head
   ↓
Logits                           (batch, seq, vocab_size)
```



### **1. Embedding的作用**

Embedding在整个作业中最核心的作用是：**连接Tokenizer和Transformer， 通过查表把这些离散的token ID 转化成可训练的稠密隐藏向量，使Transformer能够对其进行矩阵运算，并在训练过程中学习token的特征表示。**

------

##### 1. Tokenizer 的输出

输入的原始文本首先经过 Tokenizer，被转换成一串离散的 Token ID：

```
原始文本
    ↓ Tokenizer
[12, 35, 8, 21, 7, 46]
-------------------------------------
位置：       0    1   2    3   4   5
Token ID： [12, 35,  8,  21,  7, 46]

这里的数字只是每个 token 在词表中的编号。例如：
12 → 词表中的第 12 个 token
35 → 词表中的第 35 个 token
8  → 词表中的第 8 个 token

不能仅根据 ID 判断对应的文字是什么，必须查看 Tokenizer 的词表。
这些数字只是“地址”，用于告诉 Embedding 去哪一行查找向量。
```

组成 batch 后，`token_ids` 通常是一个二维整数张量：

```
shape = (batch_size, seq_len)
```

例如：

```
token_ids = torch.tensor([
    [12, 35, 8],
    [21, 7, 46],
])
```

它的形状是：

```
(2, 3)
```

其中：

- `2` 表示 batch 中有两条序列。
- `3` 表示每条序列包含三个 token。

Token ID 只是 token 在词表中的编号。例如，ID 为 `100` 的 token 并不比 ID 为 `10` 的 token“更大”，二者之间的数值距离也不表示语义距离。

因此，Token ID 不能直接作为有意义的连续数值特征参与 Transformer 的矩阵运算。它的作用只是告诉模型：需要从 Embedding Table 中取出哪一行向量。

------

##### 2. 创建 Embedding Table

Embedding 初始化时会创建一个可训练的 Embedding Table：

```
Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
)
```

对应的参数是：

```
self.weight
```

它的形状为：

```
(vocab_size, d_model)
```

其中：

- `num_embeddings = vocab_size`：词表中一共有多少个 token，也就是 Embedding Table 有多少行。
- `embedding_dim = d_model`：每个 token 对应的隐藏向量有多长，也就是 Embedding Table 有多少列。(原来的每个token值被扩展成多少维向量)

假设：

```
vocab_size = 50
d_model = 4
```

这表示：

- 词表中有 50 个 token，Token ID 范围为 `0～49`。
- 每个 token 要转换成一个长度为 4 的隐藏向量。

于是 Embedding 会创建：

```
embedding = Embedding(
    num_embeddings=50,
    embedding_dim=4,
)
```

内部的 Embedding Table 为：

```
self.weight
```

形状是：

```
(50, 4)
```

也就是：

```
50 行 × 4 列
```

可以将它想象成：

```
Token ID 0  → [4个数]
Token ID 1  → [4个数]
Token ID 2  → [4个数]
...
Token ID 49 → [4个数]
```

每一行对应一个 Token ID，每一行都是一个长度为 `d_model=4` 的隐藏向量。

可以把 Embedding Table 理解成：

```
Token ID       对应的隐藏向量

0              [0.12, -0.31, 0.08, 0.02]
1              [0.45,  0.17, -0.26, 0.38]
2              [-0.11, 0.52, 0.33, 0.56]
...
vocab_size-1   [0.27, -0.14, 0.61, 0.85]
```

每一行对应一个 Token ID，每一行都包含 `d_model` 个数。

------

##### 3. 随机初始化 Embedding Table

模型刚创建时，还不知道每个 token 应该用什么向量表示，因此会按照规定的分布为 Embedding Table 随机初始化数值。

在 CS336 的实现中，可以使用截断正态分布初始化：

```
nn.init.trunc_normal_(
    self.weight,
    mean=0.0,
    std=1.0,
    a=-3.0,
    b=3.0,
)
```

因此，模型刚初始化时，Embedding Table 中的向量只是随机数，还没有学到明确的语义。

由于 `self.weight` 被注册成了：

```
nn.Parameter
```

所以它是模型的可训练参数。训练过程中，反向传播会不断更新这些数值，使每个 token 的隐藏向量逐渐学习到对当前语言模型任务有用的特征。

因此，Embedding Table 的变化过程是：

```
随机初始化的向量
        ↓ 模型训练
不断更新的向量
        ↓
学习到有用特征的 token 表示
```

假设我们需要用到的几行被随机初始化成：

| Embedding Table 行 | 隐藏向量                     |
| ------------------ | ---------------------------- |
| `weight[7]`        | `[0.41, -0.22, 0.09, 0.73]`  |
| `weight[8]`        | `[-0.16, 0.30, -0.71, 0.12]` |
| `weight[12]`       | `[0.20, -0.40, 0.10, 0.80]`  |
| `weight[21]`       | `[0.55, -0.18, 0.24, -0.31]` |
| `weight[35]`       | `[-0.50, 0.60, 0.30, -0.20]` |
| `weight[46]`       | `[0.14, 0.27, -0.43, 0.66]`  |

这些数值只是为了演示而假设的随机数。

------

##### 4. `forward()` 的作用：查表

Embedding 的 `forward()` 可以写成：

```
def forward(self, token_ids):
    return self.weight[token_ids]
```

这里的：

```
self.weight[token_ids]
```

表示把 `token_ids` 中的每个整数作为 Embedding Table 的行索引，取出对应的整行向量。

例如：

```
token_ids = torch.tensor([12, 35, 8, 21, 7, 46])
```

那么：

```
self.weight[token_ids]
```

等价于：

```
torch.stack([
    self.weight[12],
    self.weight[35],
    self.weight[8],
    self.weight[21],
    self.weight[7],
    self.weight[46],
])
```

具体对应关系是：

```
Token ID 2 → 取出 Embedding Table 的第 2 行
Token ID 5 → 取出 Embedding Table 的第 5 行
Token ID 8 → 取出 Embedding Table 的第 8 行
```

因此，`forward()` 的整个过程本质上就是查表。

注：PyTorch 会按照 `token_ids` 中的顺序取出对应的行，而不是按照 ID 的大小排序。

------

##### 5. 每个整数被替换成一个向量

`embedding_dim`（在 Transformer 中通常等于 `d_model`）规定了每个 token 隐藏向量的长度。

Embedding 的 `forward()` 会把 `token_ids` 中每个位置上的一个整数 Token ID，映射为 Embedding Table 中对应的一行，也就是一个长度为 `d_model` 的隐藏向量。

具体的查表过程如下：

| 序列位置 | Token ID | 查找的行     | 得到的隐藏向量               |
| -------- | -------- | ------------ | ---------------------------- |
| 0        | 12       | `weight[12]` | `[0.20, -0.40, 0.10, 0.80]`  |
| 1        | 35       | `weight[35]` | `[-0.50, 0.60, 0.30, -0.20]` |
| 2        | 8        | `weight[8]`  | `[-0.16, 0.30, -0.71, 0.12]` |
| 3        | 21       | `weight[21]` | `[0.55, -0.18, 0.24, -0.31]` |
| 4        | 7        | `weight[7]`  | `[0.41, -0.22, 0.09, 0.73]`  |
| 5        | 46       | `weight[46]` | `[0.14, 0.27, -0.43, 0.66]`  |

因此：

```
12 → weight[12]
35 → weight[35]
 8 → weight[8]
21 → weight[21]
 7 → weight[7]
46 → weight[46]
```

这就是所谓的“把 Token ID 当作行索引”。

只是在做：

```
看到 ID 12 → 取出第 12 行
```

需要注意：Embedding 并不会修改原来的 `token_ids` 张量，而是根据它创建一个新的浮点数张量。

------

##### 6. 得到 Embedding 输出

最终输出为：

```
[
    [ 0.20, -0.40,  0.10,  0.80],  # token_id 12
    [-0.50,  0.60,  0.30, -0.20],  # token_id 35
    [-0.16,  0.30, -0.71,  0.12],  # token_id 8
    [ 0.55, -0.18,  0.24, -0.31],  # token_id 21
    [ 0.41, -0.22,  0.09,  0.73],  # token_id 7
    [ 0.14,  0.27, -0.43,  0.66],  # token_id 46
]
```

输入中原来每个位置只有一个整数：

```
[12, 35, 8, 21, 7, 46]
```

现在每个位置都对应一个长度为 `d_model=4` 的向量：

```
[
    向量12,
    向量35,
    向量8,
    向量21,
    向量7,
    向量46,
]
```

这就是：

> 把 `token_ids` 中每个位置上的一个整数 Token ID，映射成一个长度为 `d_model` 的隐藏向量。

##### 7. Embedding 前后的形状变化

假设输入的形状是：

```
(batch_size, seq_len)
```

输入中每个位置原来只有一个整数 Token ID。

Embedding 把每个整数转换成一个长度为 `d_model` 的向量，因此输出形状是：

```
(batch_size, seq_len, d_model)
```

也就是：

```
(B, S) → (B, S, D)
```

其中：

- `B`：batch size。
- `S`：序列长度。
- `D`：`d_model`。

例如：

```
输入 shape： (1,6)
输出 shape： (1,6,4)
```

这表示：

- 仍然有1条序列。
- 每条序列仍然有6个 token 位置。
- 但每个位置不再是一个整数，而是一个包含 4 个浮点数的隐藏向量。

所以形状变化是：

```
(1,6,) → (1,6, 4)
```

因此，更准确的说法是：

> Embedding 保留了 `token_ids` 原有的 batch、序列长度、token 顺序和位置对应关系，并在最后增加一个 `d_model` 维度；它没有保留 Token ID 本身的数值含义，而是把每个位置上的整数 Token ID 映射成一个长度为 `d_model` 的隐藏向量。

------

##### 8. 相同 Token ID 会查到相同的向量

例如：

```
token_ids = torch.tensor([12, 35, 8, 21, 7,46])
```

查表结果是：

```
[
    weight[12],
    weight[35],
    weight[8],
    weight[21],
    weight[7],
    weight[46],
]
```

两个 Token ID `2` 都会取出同一行 `weight[2]`。

所以在刚经过 Token Embedding 时，相同的 Token ID 会得到相同的初始隐藏向量。

但是，当这些向量继续经过 RoPE、自注意力和 Transformer Block 后，即使是相同的 token，由于所在位置和上下文不同，最终的隐藏状态也可能不同。

------

##### 9. Embedding 保留顺序，但不负责理解位置

Embedding 会保持 token 原来的排列关系：

```
[12, 35, 8, 21, 7, 46]
```

经过查表后仍然按照相同顺序排列：

```
[
  weight[12], 
  weight[35], 
  weight[8], 
  weight[21], 
  weight[7], 
  weight[46]
]
```

但是，Embedding 本身只是根据 Token ID 查表，并不知道一个 token 位于序列的第几个位置。

也就是说：

- Embedding 保留了位置对应关系。
- Embedding 本身不编码绝对或相对位置信息。
- CS336 模型之后使用 RoPE，让注意力计算能够感知 token 的位置关系。

------

## 2. Linear的作用

`Linear` 可以理解成模型里的**“信息加工器”或“变换器”**。

`Linear` 在 Transformer 里的核心作用，可以概括成一句话：

> 用一组可训练的权重，对隐藏向量中的特征重新组合，并根据需要改变向量维度。

`Linear` 是 Transformer 中**可重复使用的可学习特征投影工具**。它通过权重矩阵对每个 token 的隐藏向量进行特征重新组合，并根据模块需要保持、扩大或缩小最后一个维度。不同位置的 Linear 拥有不同的独立权重，这些权重会通过反向传播学习适合当前位置的特征转换方式。

在 Self-Attention 中，三个独立的 Linear 将同一个 token 隐藏向量分别投影成 Q、K、V。Q 表示当前 token 希望寻找的信息，K 表示当前 token 可供匹配的特征，V 表示匹配成功后实际传递的内容。Linear 只负责生成 Q、K、V，真正的 token 匹配和信息聚合由后面的 Attention 运算完成。

在 SwiGLU 中，`w1` 和 `w3` 分别把每个 token 的隐藏向量从 `d_model` 扩展到 `d_ff`：`w1` 产生门控特征，`w3` 产生内容特征；经过 `SiLU(w1(x)) * w3(x)` 完成特征筛选和组合，再由 `w2` 压缩回 `d_model`。这个过程对每个 token 独立进行非线性特征加工。

注意：

> Linear 的权重矩阵中的具体数值是训练学出来的。不同位置的 Linear 因为承担的功能不同，会通过反向传播学出不同的权重矩阵。

- 权重矩阵的形状由设计者规定
- 权重矩阵中的数值由模型训练得到

> Linear 权重矩阵的形状和所在位置由模型架构决定，但矩阵内部的具体数值不是人工设定的，而是先随机初始化，再通过损失函数、反向传播和优化器共同训练得到。不同位置的 Linear 通常拥有相互独立的权重，因此会根据各自在模型中的作用，学习不同的特征转换方式。

唯一的例外是模型明确进行参数共享。例如 tied embeddings 会让输入 Embedding 和输出 LM Head 共用同一套权重。除此以外，即使两个 Linear 的形状完全相同，默认也会学习出不同的权重。

## 3.RMSNorm 

**RMSNorm 的全称是 Root Mean Square Normalization，即均方根归一化。**

RMSNorm 可以理解为 Transformer 中一个带可学习参数的隐藏向量尺度调节器。

RMSNorm 的核心作用是：

> **在隐藏向量进入 Attention、SwiGLU 或最终输出层之前，稳定每个 token 隐藏向量的整体数值尺度，使后续计算和模型训练更加稳定。**

它不会改变张量形状，也不会像 `Linear` 那样混合不同特征。它主要做的是“控制向量整体大小”。

RMSNorm 的计算公式是：
$$
\operatorname{RMSNorm}(x) = \frac{x} {\sqrt{\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}x_i^2+\epsilon}} \odot g 
$$
其中：

- `x`：一个 token 的隐藏向量
- `d_model`：隐藏向量的特征数量
- `ε`：防止除以零的小常数
- `g`：可训练的缩放参数，也就是代码中的 `self.weight`
- `⊙`：逐元素相乘

它完成两件事：

1. 用隐藏向量的 RMS 控制整体尺度。
2. 用可训练的 `weight` 调整每个特征的重要程度。

##### 1. 为什么需要RMSNorm

Transformer 在层与层之间不断执行：

- Attention
- Linear
- SwiGLU
- 残差相加

经过很多层以后，隐藏向量可能变得很大或很小。

例如某一层输入：

```
[100, 200, 300, 400]
```

另一层输入：

```
[0.001, 0.002, 0.003, 0.004]
```

虽然这两个向量内部的比例关系完全一样，但数值尺度差别非常大。这容易造成：

- 梯度不稳定
- 激活值越来越大或越来越小
- BF16/FP16 下发生数值精度问题
- 模型训练速度变慢
- 深层 Transformer 难以优化

RMSNorm 会把它们调整到接近相同的整体尺度。

注意，它不是把所有数字变得一样，而是保留它们的相对大小。

##### 2. RMSNorm在 Transformer 中的三个使用位置

CS336 使用的是 Pre-Norm Transformer。

在 Pre-Norm Transformer 中，它主要出现在三类位置：

1. 在 Self-Attention 之前，稳定生成 Q、K、V 所使用的隐藏向量。
2. 在 SwiGLU 之前，稳定进入 `w1`、`w3` 门控分支的隐藏向量。
3. 在全部 Transformer Block 之后，稳定进入 `lm_head` 的最终隐藏向量。

最核心的两行代码是：

```
x = x + attention(rmsnorm1(x))
x = x + swiglu(rmsnorm2(x))
```

其中：

- RMSNorm 负责稳定尺度；
- Attention 负责 token 之间的信息交换；
- SwiGLU 负责对每个 token 的特征进行非线性加工；
- 残差连接负责保留原有信息并提供稳定的梯度路径；
- 最终 RMSNorm 负责整理进入语言模型输出层的最终隐藏状态。

如果模型有 `N` 个 Transformer Block：

- 每个 Block 有两个 RMSNorm；
- 所有 Block 后还有一个最终 RMSNorm。

所以总数通常为：2N+1 

例如 8 层 Transformer 一共有：
$$
2\times8+1=17 
$$
个独立的 RMSNorm 模块。这些 RMSNorm 结构相同，但各自拥有独立的 `weight` 参数。

## 4. SiLU()函数

 `silu()` 在 Transformer 中的核心作用是：作用于前馈神经网络，**给线性变换后的隐藏向量加入非线性**，并在 SwiGLU 中参与控制哪些特征应该被保留、削弱或增强。属于激活函数的一种。

它通常不直接用于 Self-Attention，而是用于 Transformer Block 的 FFN/SwiGLU 部分。SiLU 主要用于实现 `SwiGLU`：

SwiGLU(x)=W2(SiLU(W1x)⊙W3x)

## 5. SwiGLU

**FNN本质**：先把输入的内容进行升维，展开足够多的细节，这是为了进行深度思考；然后进行非线性激活，在高维空间中，通过激活函数处理数据，这是在进行真正的思考和筛选；最后是降维，把处理好的信息压缩回原来的维度，即把思考的精华总结输出

**SwiGLU 本质上就是 Transformer 前馈神经网络 FFN 的一种门控变体。**

更准确地说：

> SwiGLU 不是 Transformer 之外新增的第三个主要模块，而是用来替代传统 FFN 的一种更强的前馈网络结构。

它的主要作用是：

> 对每个 token 的隐藏向量进行更深层的特征加工，选择有用信息、组合不同特征，然后重新输出一个长度为 `d_model` 的隐藏向量。

```
数据流为：

              ┌─ w1 → SiLU ── gate ─┐
输入 x ───────┤                      ├─ 逐元素相乘 → w2 → 输出
              └─ w3 ──────── value ─┘
                      d_ff
```

1. `w1`：生成门控特征

首先，`w1` 把隐藏向量从 `d_model` 扩展到 `d_ff`，然后经过 SiLU非线性调节，生成**门控调节信号**

核心作用是：学习当前 token 的每个中间特征应该被削弱、保留、增强还是改变符号。

2. `w3`：生成待被加工的内容特征

`w3` 同样把输入从 `d_model` 扩展到 `d_ff`，但 `w3` 和 `w1` 有不同的独立权重。w3负责生成**等待被门控和加工的内容特征**

3. `w1` 和 `w3` ：用调节信号处理内容

   `w1` 和 `w3` 并行接收同一个 token 隐藏向量，经过各自处理后，两个分支的结果逐元素相乘，使模型能够动态地削弱、保留或增强不同中间特征。

4. `w2`：整合结果并恢复维度

`w2` 把 `d_ff` 个中间特征重新组合，并压缩回 `d_model`，以便与残差分支相加。`w2` 不只是简单删除维度，而是为每个输出维度学习一组权重，对所有中间特征进行加权组合。

```
Transformer Block
├── Self-Attention：token 之间交换信息
└── FFN（CS336 中使用 SwiGLU）：token 内部加工信息
```

## 6. Softmax

`Softmax` 是一个不包含可学习参数的函数。它沿指定的 `dim` 维度，将任意实数分数转换成一组**非负且总和为 1** 的归一化权重。

因此，Softmax 本质上是一个：

> 将原始分数转换成权重或概率的“比例分配器”。

Softmax公式为：
$$
\operatorname{Softmax}(x_i) = \frac{e^{x_i}} {\sum_j e^{x_j}} 
$$
为了防止指数运算发生数值溢出，实际代码使用数值稳定版本：
$$
\operatorname{Softmax}(x_i) = \frac{e^{x_i-\max(x)}} {\sum_j e^{x_j-\max(x)}} 
$$
所有元素同时减去最大值，只改变整体位置，不会改变最终 Softmax 结果。

#### 1. Softmax 在 Transformer 中的作用

Softmax 在 Transformer 中主要出现在两个地方：

##### 1.1Self-Attention：把 token 之间的匹配分数转换成注意力权重。

Scaled Dot-Product Attention(即注意力) 的公式是：
$$
\operatorname{Attention}(Q,K,V) = \operatorname{Softmax} \left( \frac{QK^\top}{\sqrt{d_k}}+\text{mask} \right)V
$$
在 Transformer 的 Self-Attention 中，Softmax 将**经过缩放和 causal mask 处理**的 Q、K 匹配分数转换成 Attention 权重，使每个 query 能够按照相关程度对所有可见 token 的 V 进行加权求和。

```
隐藏向量
    ↓
三个独立的 Linear
    ↓
生成 Q、K、V
    ↓
QKᵀ 计算 token 之间的匹配分数
    ↓
除以 √dₖ 控制分数尺度
    ↓
Causal Mask 屏蔽未来 token
    ↓
Softmax 把分数转换成注意力权重
    ↓
使用注意力权重对 V 加权求和
    ↓
获得融合上下文信息的新隐藏向量
```

各部分的分工如下：

- Q：表示当前 token 想寻找什么信息。
- K：表示每个 token 可用于匹配的特征。
- `QKᵀ`：计算 query 与各个 key 的匹配分数。
- 缩放：防止分数过大，使 Softmax 过于极端。
- Mask：把不允许关注的位置设为负无穷。
- Softmax：把匹配分数转换成总和为 1 的权重。
- V：提供被加权和汇总的实际内容。

因此可以概括为：

> Q 和 K 决定“应该关注谁”，Softmax 决定“分别关注多少”，V 决定“真正获取什么信息”。

##### 1.2 文本生成：把词表 logits 转换成下一个 token 的概率。

在文本生成中，Softmax 将最后一个位置的词表 logits 转换成下一个 token 的概率分布，供 temperature、top-p 和随机采样使用。但在模型训练时，Transformer 应直接输出 logits，再交给交叉熵损失函数，不应提前执行 Softmax。

注：**Logits 是模型经过最后一个 Linear 层后，对词表中每个 token 给出的未归一化原始分数。Logit 越大，模型认为对应 token 越适合作为下一个 token。Logits 本身不是概率，经过 Softmax 后才会转换为总和为 1 的概率分布。**

在 Transformer 语言模型中：

```
token IDs
   ↓
Embedding
   ↓
Transformer Blocks
   ↓
最终隐藏向量
   ↓
lm_head（Linear）
   ↓
logits
   ↓
Softmax
   ↓
下一个 token 的概率
```

**`lm_head` 是 Transformer 语言模型最后的词表预测层。它通常是一个 `Linear(d_model, vocab_size)`，负责把每个位置的隐藏向量转换为词表中所有 token 的 logits。随后可以通过 Softmax 将 logits 转换为概率，从而预测或采样下一个 token。**

## 7. RoPE（Rotary Position Embedding，旋转位置编码）

Transformer 中常见的位置编码包括**绝对位置编码**和**相对位置编码**两大类。Rope 旋转位置编码就是属于相对位置编码，是一种经常被用在大模型（比如LLaMA、chatGLM）里面的方式。

##### 1.为什么要使用位置编码

Embedding 主要表示 token 的内容。同一个 token 无论出现在第几个位置，最初查到的 Embedding 向量都是相同的。

如果没有位置相关信息，普通自注意力主要根据 token 的内容计算相关性。当“你喜欢狗”变成“狗喜欢你”时，Attention 看到的仍然是“你、喜欢、狗”这些内容，只是排列发生了变化，因此不能充分理解谁在前、谁在后以及谁喜欢谁。

位置编码就是给 token 加入位置信息，使模型在关注 token 内容的同时，还能理解 token 的先后顺序和相对距离。

在 Llama 中，RoPE 根据 token 的位置旋转 $Q$ 和 $K$，使注意力分数同时包含：

```
token 的内容关系
+
token 的相对位置关系
```

而 Causal Mask 负责禁止模型查看未来 token。

举例：

###### 1. Tokenizer 和 Embedding 做了什么

假设分词结果是：

```
你喜欢狗 → [你, 喜欢, 狗]
狗喜欢你 → [狗, 喜欢, 你]
```

Tokenizer 把它们转换成 Token ID：

```
你喜欢狗 → [10, 25, 38]
狗喜欢你 → [38, 25, 10]
```

Embedding 再根据 Token ID 查表：

```
“你”   → 向量 A
“喜欢” → 向量 B
“狗”   → 向量 C
```

所以：

```
你喜欢狗 → [A, B, C]
狗喜欢你 → [C, B, A]
```

重点是：

> 同一个 token 无论放在什么位置，最初查到的 Embedding 向量都一样。

因此，“你”放在第一个位置时是向量 A，放在第三个位置时，仍然是向量 A。

###### 2. 没有位置编码时，Attention 看到了什么？

先暂时不考虑 Causal Mask。

对于“你喜欢狗”：

```
“喜欢”会和：
你
喜欢
狗

分别计算相关性。
```

对于“狗喜欢你”：

```
“喜欢”还是会和：
狗
喜欢
你

分别计算相关性。
```

虽然排列顺序变了，但“喜欢”看到的仍然是同样三个内容：

```
你、喜欢、狗
```

又因为这些 token 的向量没有变化，所以“喜欢”与它们计算出来的内容相关性也不会因为前后顺序发生变化。

Attention 此时只能知道：

```
句子里有“你”“喜欢”“狗”
```

却不知道：

```
是“你”在“喜欢”前面
还是“狗”在“喜欢”前面
```

因此

> 相同 token 得到的处理结果基本相同，只是这些结果随着 token 的顺序一起被重新排列。

例如：

```
你喜欢狗 → [处理后的你, 处理后的喜欢, 处理后的狗]

狗喜欢你 → [处理后的狗, 处理后的喜欢, 处理后的你]
```

这就无法充分表达“谁喜欢谁”。

###### 3. 加入位置编码以后

普通位置编码可以理解为：给每个座位分配一个不同的“座位编号”。

```
第一个位置 → 位置1
第二个位置 → 位置2
第三个位置 → 位置3
```

于是“你喜欢狗”变成：

```
你   的内容向量 + 位置1
喜欢 的内容向量 + 位置2
狗   的内容向量 + 位置3
```

“狗喜欢你”变成：

```
狗   的内容向量 + 位置1
喜欢 的内容向量 + 位置2
你   的内容向量 + 位置3
```

虽然“你”的内容向量还是同一个 $A$，但是：

```
你在第一个位置：A + 位置1
你在第三个位置：A + 位置3
```

这两个表示就不再相同了。

模型因此能够知道：

```
“你”出现在“喜欢”之前
“狗”出现在“喜欢”之后
```

于是可以区分：

```
你喜欢狗
狗喜欢你
```

###### 4. Llama 中的 RoPE 怎么做？

Llama 不会直接把位置向量加到 Embedding 上。

它先生成：

```
X
↓
Q、K、V
```

然后根据每个 token 的位置，对 $Q$ 和 $K$ 进行不同角度的旋转：

```
第1个位置 → 按位置1旋转
第2个位置 → 按位置2旋转
第3个位置 → 按位置3旋转
```

于是两个 token 计算注意力分数时，不仅受到内容影响，还会受到它们之间位置关系的影响。

例如模型可以区分：

```
“你”在“喜欢”前面1个位置
“你”在“喜欢”后面1个位置
“你”距离“喜欢”10个位置
```

因此，RoPE 主要把“相对顺序和相对距离”加入 Attention。

###### 5. Causal Mask 和位置编码不是一回事

Llama 还会使用 Causal Mask。

它的作用是：

```
第1个 token：只能看自己
第2个 token：可以看第1、2个
第3个 token：可以看第1、2、3个
```

所以在真实的 Decoder-only Transformer 中，即使没有 RoPE：

```
你喜欢狗
狗喜欢你
```

得到的结果通常也不会完全相同，因为每个位置能看到的前文不同。

但 Causal Mask 主要告诉模型：

> 只能看自己和前面的 token，不能看未来。

RoPE 则进一步告诉模型：

> 某个 token 在我的前面还是后面，以及它距离我多远。

可以记成：

```
Causal Mask：规定“可以看谁”
RoPE：说明“对方在哪里”
```

## 8. Scaled Dot-Product Attention（缩放点积注意力）

Scaled Dot-Product Attention 是 Transformer 中负责 token 间信息交流的核心运算。它先使用 Q 和 K 的点积计算每个 query token 与各 key token 的相关性，再除以 sqrt{d_k} 防止分数过大；Decoder-only 模型随后使用 Causal Mask 屏蔽未来位置，并通过 Softmax 将分数转换成注意力权重，最后按照这些权重对 V 做加权求和，得到包含上下文信息的新隐藏向量。Multi-Head Self-Attention 会在多个 Head 中并行执行这一运算，以学习不同类型的 token 关系。

Scaled Dot-Product Attention 是连接多部分内容的核心：

- `Linear`：生成 Q、K、V。
- `RoPE`：为 Q、K 注入位置信息。
- `Softmax`：把匹配分数变成注意力权重。
- `Causal Mask`：禁止看到未来。
- `Multi-Head Attention`：并行执行多组 Scaled Dot-Product Attention。
- `Transformer Block`：通过残差连接保存并加入 Attention 信息。
- `TransformerLM`：堆叠多层，让 token 反复交换和加工信息。
- `Cross Entropy`：根据最终 logits 计算误差，并通过反向传播训练 Q/K/V 投影等参数。

在 CS336 Systems/FlashAttention 部分，优化的仍然是同一套数学运算：
$$
\operatorname{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$
FlashAttention 没有改变 Attention 的含义，只是使用分块和在线 Softmax，避免完整保存巨大的 `seq_len × seq_len` 注意力矩阵，从而降低显存访问和中间内存开销。





















