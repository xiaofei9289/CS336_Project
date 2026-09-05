## BPE部分

bpe.py, train_bpe.py和tokenizer.py三者共同实现了一套完整的byte-level BPE Tokenizer

- bpe.py：提供训练需要的底层函数。
- train_bpe.py：组织完整的 BPE 训练循环。
- tokenizer.py：使用训练结果处理新文本，并编码和解码

> `bpe.py` 是 BPE 训练的“零件库”，`train_bpe.py` 是把这些零件组装起来的“训练控制器”，`tokenizer.py` 则负责使用训练结果进行实际的编码和解码。

#### 步骤一：完成bpe.py

整个文件的作用是：提供训练 Byte-level BPE 分词器所需的底层工具函数。

它不负责完整的训练循环，也不负责最终文本的编码与解码。它主要完成以下四件事：

- 创建初始词表
- 把训练文本的预分词转化成byte ID
- 统计并选择最值得合并的相邻token pair
- 把选中的pair合并成新的token

```
原始文本
  ↓
去掉特殊 token 边界
  ↓
GPT-2 正则预分词
  ↓
转换成 byte ID 序列
  ↓
统计相邻 pair
  ↓
选择频次最高的 pair
  ↓
合并成新 token
  ↓
重复统计和合并，直到达到目标词表大小
```

```python
GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

TokenIds = tuple[int, ...]
TokenPair = tuple[int, int]
PretokenCounts = dict[TokenIds, int]
```

##### 函数1：初始化基础词表

```python
def initialize_vocab(
    vocab_size: int,
    special_tokens: list[str],
) -> dict[int, bytes]:
```

作用：创建BPE训练前的基础词表

1. 建立 256 个基础 byte token。
2. 加入特殊 token，并确保 ID 不重复。
3. 确认初始词表大小是 `256 + 特殊 token 数量`。
4. 暂时不要优化，也不要并行化。

生成的vocab有两个用途：

- `select_best_pair()` 用它比较 pair 对应的实际 bytes；
- 训练循环用它创建合并后的新 token。

##### 函数2：对文本按special token进行分割，并把特殊token从结果中删除

```python
def split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
```

为什么要这么做：因为特殊token通常表示文档边界，如果不切开，BPE可能训练了跨越边界的pair

作用：把文本按special_token进行分割

1. 特殊token去重
2. 没有特殊token直接返回原文本
3. 禁止空字符串
4. 设置特殊长token优先
5. 转义正则特殊字符
6. 切开文本

##### 函数3：对文本进行预分词

```python
def pretokenize(
    text: str,
    special_tokens: list[str],
) -> PretokenCounts:
```

作用：把用来训练的文本转换成{byte ID 序列 → 出现次数}，这是后续BPE pair统计使用的主要数据结构

1. 创建计数器
2. 按照特殊token切分
3. 使用GPT-2预分词把普通文本片段切成pre-token
4. 转换成UTF-8 bytes
5. 把bytes转化成整数tuple
6. 统计频次
7. 检查正则有没有遗漏文本

##### 函数4：统计所有pre-token内部相邻的token pair的全局出现次数

```python
def count_adjacent_pairs(
    pretoken_counts: dict[tuple[int, ...], int],
) -> dict[TokenPair, int]:
```

作用：只统计单个 pre-token 内部的 pair，不会跨 pre-token 统计。保证 BPE 合并不会跨越 GPT-2 预分词边界。

##### 函数5：从所有相邻pair中选择本轮要合并的最佳pair

```python
def select_best_pair(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> TokenPair | None:
```

选择规则：

1. 优先选择出现频次最高的pair
2. 如果频次相同，选择对应byte值字典序更大的pair

##### 函数6：在一个pre-token内，从左到右把目标pair合并成新的token ID

```python
def merge_pairs(
    token_ids: tuple[int, ...],
    target_pair: tuple[int, int],
    new_token_id: int,
) -> TokenIds:
```

##### 函数7：把本轮选中的 pair 应用到所有 pre-token，并生成新的频次表。

```python
def merge_all_pretokens(
    pretoken_counts: PretokenCounts,
    pair_to_merge: TokenPair,
    new_token_id: int,
) -> PretokenCounts:
```

------

#### 文本预处理分支(bpe.py)：

```
stage: 输入文本→经过pretokenize()函数→文本转换成{byte ID 序列：出现次数}
```

其中，pretokenize()函数过程：

```
stage1: 输入文本→经过split_on_special_tokens()函数→得到被特殊token分开的多个普通文本片段
注：特殊 token 本身不保留
过程举例：
起始：
text = "Hello<|endoftext|>world"
special_tokens = ["<|endoftext|>"]
得到：
["Hello", "world"]
```

```
stage2: 单独处理每个文本片段→经过GPT-2正则预分词规则后→得到若干pre-token
注：每个pre-token本质都是一个 Python 字符串
过程举例：
起始：
"Hello world! 你好"
得到：
[
  "Hello"   # str
  " world"  # str，包含开头空格
  "!"       # str
  " 你好"    # str，包含开头空格
]
```

```
stage3: 每个pre-token字符串→经过utf-8编码后→pretoken_bytes
本质上是：bytes sequence
过程举例：
"Hello".encode("utf-8") → # b'Hello'

" world".encode("utf-8") → # b' world'

"!".encode("utf-8") → # b'!'

" 你好".encode("utf-8") → # b' \xe4\xbd\xa0\xe5\xa5\xbd'
```

```
stage4: pretoken_bytes→经过tuple(pretoken_bytes)→得到字节整数元组tuple[int, ...]

过程举例：
tuple(b"Hello") → (72, 101, 108, 108, 111)
tuple(b" world") → (32, 119, 111, 114, 108, 100)
tuple(b"!") → (33,)
```

```
stage5: 每个 pre-token 对应的字节整数元组→经过Counter 计数→得到 {字节整数元组: 出现次数}
即：pretoken_counts，{token_ids:出现次数}
```

**完整过程**

```
原始文本
str
"Hello world! 你好"

        ↓ split_on_special_tokens()

普通文本片段
list[str]
["Hello world! 你好"]

        ↓ GPT-2 正则预分词

pre-token 字符串
str
"Hello"
" world"
"!"
" 你好"

        ↓ encode("utf-8")

UTF-8 bytes
bytes
b"Hello"
b" world"
b"!"
b" \xe4\xbd\xa0\xe5\xa5\xbd"

        ↓ tuple()

字节整数元组
tuple[int, ...]
(72, 101, 108, 108, 111)
(32, 119, 111, 114, 108, 100)
(33,)
(32, 228, 189, 160, 229, 165, 189)

        ↓ Counter 计数

pre-token 频次表
Counter[tuple[int, ...]]
{
    (72, 101, 108, 108, 111): 1,
    (32, 119, 111, 114, 108, 100): 1,
    (33,): 1,
    (32, 228, 189, 160, 229, 165, 189): 1,
}
```

------

##### bpe训练分支

此时进入train_bpe.py，该文件负责组织训练循环，并调用 `bpe.py` 中的底层函数

```
stage1: 初始化词表     
调用initialize_vocab()函数初始化基础词表, 包含256个基础 byte token和特殊 token 的初始词表 vocab。
```

```
stage2: 预处理训练语料    
从磁盘中读取训练语料，调用pretokenize()函数后，经过文本预处理，得到pretoken_counts
pretoken_counts 的格式是：
{
    token ID序列: 出现次数
}
```

##### 单轮 BPE Merge

```
stage3: 统计相邻 pair    
pretoken_counts → 经过count_adjacent_pairs()函数 → pair_counts
将 pretoken_counts 输入 count_adjacent_pairs()，根据每个 pre-token 的出现频次，加权统计所有相邻 pair，得到 pair_counts。
```

```
stage4:选择最佳 pair       
把pair_counts和vocab同时输入到select_best_pair()函数，经过比较后得到频次最高的pair,即best pair

比较原则：
优先选择出现频次最高的 pair；
频次相同时，比较两个 token 在 vocab 中对应的 bytes，选择 bytes 字典序更大的 pair。
```

```
stage5: 
为 best_pair 分配 new_token_id；
将两个旧 token 对应的 bytes 拼接起来并加入 vocab；
同时将本轮合并规则记录到 merges。
```

```
stage6: 把合并规则应用到全部 pre-token
调用merge_all_pretokens()函数，遍历频次表中的所有 pre-token；针对每一个 pre-token 调用内部的 merge_pairs()，从左到右把每个 pre-token 中所有非重叠的 best_pair 替换为 new_token_id，并保留对应的出现频次,生成新的 pretoken_counts。
```

```
stage7:
使用更新后的 pretoken_counts 进入下一轮 pair 统计。重复 Stage 3～Stage 6，直到词表达到 vocab_size，或者已经不存在可合并的 pair。
```

```
stage8: 最终输出vocab 和 merges。
```

------

##### `tokenizer.py`：使用训练结果编码和解码

 `Tokenizer` 不负责训练，它接收 `train_bpe()` 产生的：vocab和merges，然后用于：

- 文本编码：`str -> list[int]`
- token解码：`list[int] -> str`

##### 1.`Tokenizer.__init__()`：初始化编码器

它主要建立四种数据：

1.保存训练结果

```python
self.vocab = dict(vocab)
self.merges = list(merges)
```

2.建立反向词表

```python
self.inverse_vocab = self._build_inverse_vocab()
```

3.建立 merge 优先级

```python
self.merge_ranks = self._build_merge_ranks()
```

4.建立特殊 token 映射

```python
self.special_token_ids[special_token] = token_id
```

5.编译特殊 token 正则

```python
self._special_token_pattern = regex.compile(...)
```

##### 2.encode()

```python
def encode(self, text: str) -> list[int]:
```

作用：把完整文本编码成 token ID 列表。

```
原始文本
→ 区分普通文本和特殊 token
→ 特殊 token 直接映射为 ID
→ 普通文本进行 GPT-2 预分词
→ 转成单字节 token ID
→ 应用训练好的 BPE merges
→ 得到最终 token ID
```

调用关系

```
encode()
├── _split_on_special_tokens()
└── _encode_ordinary_text()
    ├── _encode_initial_bytes()
    └── _apply_bpe_merges()
        └── _merge_pair_once()
```

##### 3.encode_iterable()

作用：逐段编码一组字符串，并逐个产生 token ID。

##### 4.decode()

作用：把 token ID 列表解码回字符串。

```
token ID
→ 查询 vocab 得到 bytes
→ 拼接所有 bytes
→ 使用 UTF-8 解码
→ 得到字符串
```

------

`Linear` 的作用，大白话说就是：

> 把一个输入向量中的信息重新混合，生成一个新的向量。
