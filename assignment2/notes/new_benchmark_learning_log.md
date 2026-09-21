# 从零理解 benchmark：我的学习与实践记录

记录日期：2026-09-19  
对应资料：CS336 Assignment 2，Spring 2026，Version 26.1.3，第 2 章，重点是 2.1.3 End-to-End Benchmarking。  
实践文件：[new_benchmark.py](../cs336_systems/new_benchmark.py)  
参考模型：[model.py](../cs336-basics/cs336_basics/model.py)

> 本文记录本次对话中的学习过程、实际修改和运行反馈，不是完整作业答案。Windows 终端结果由我提供；代码检查为只读检查。当前已验证 GPU 前向可运行，尚未完成计时实验或全面数值正确性验证。

## 1. 为什么需要 benchmark.py

最初困惑：模型已经能够运行，为什么还要写一个 benchmark 文件？

模型负责完成计算，benchmark 负责组织和测量计算。模型返回预测结果，不会自动告诉我：这次执行花了多久、结果是否稳定、改变精度或模型规模后是否更快。

可以把 Transformer 理解为运动员，把 benchmark 理解为计时员。计时员需要明确比赛项目、安排预热、重复计时并记录结果。

本章的三条主线是：

| 方法 | 回答的问题 |
| --- | --- |
| 端到端 benchmark | 一次指定的执行需要多久？ |
| Nsight Systems profiling | 时间花在哪些操作和等待上？ |
| 显存 profiling | 什么数据在什么时候占用显存？ |

**为什么先学 benchmark：**后续讨论优化，需要一个可信的比较起点。

## 2. 先建立必要的基础概念

### 2.1 张量、形状和类型

- 张量：按一定维度组织起来的数字。
- shape：各个维度的大小，决定元素数量。
- dtype：元素的表示格式，影响存储大小和数值精度。
- device：张量位于哪个设备，例如 CPU 或第一张 CUDA GPU。

这三个属性回答不同的问题：数据怎样排列、每个元素怎样表示、数据放在哪里。

### 2.2 训练一步包含什么

| 阶段 | 做什么 |
| --- | --- |
| 前向传播 | 使用当前参数计算预测 |
| 损失计算 | 衡量预测与目标的差异 |
| 反向传播 | 计算损失对参数的梯度 |
| 优化器更新 | 根据梯度及优化器状态调整参数 |

一次前向不会自动完成后面三个阶段。

### 2.3 三个重要原理

**提交完成不等于执行完成。** CPU 可以在提交 GPU 任务后继续执行，此时 GPU 可能仍在排队或计算。计时必须考虑异步执行。

**数学运算简单不等于耗时少。** 读取、写回大量数据，以及启动和调度任务，也需要时间。

**前向结束不等于所有中间数据都能释放。** 反向计算梯度可能需要前向输入或中间结果，所以自动求导会保留必要信息；有些方法则通过重算取得这些信息。

## 3. 确定使用哪个模型

我使用作业提供的 Transformer 实现，而不是重新实现模型。

项目根目录的 `pyproject.toml` 声明了 `cs336-basics` 依赖，并把它指向本地 `./cs336-basics`，采用 editable 模式。参考语言模型类是 `BasicsTransformerLM`，位于 `cs336_basics.model` 中。

这里要区分：

1. **导入模型定义：**找到模型的设计。
2. **创建模型实例：**按配置创建具体结构和参数。
3. **运行模型：**交给它输入，执行计算并得到输出。

**为什么这么做：**性能实验应基于明确的模型实现。成功导入只是接入的第一步，不等于已经完成前向或性能验证。

## 4. 决定从哪段工作开始

第 2.1.3 节最终要求支持三种模式：

| 模式 | 工作范围 |
| --- | --- |
| Forward-only | 前向传播 |
| Forward + backward | 前向、损失和反向传播 |
| Full training step | 前向、损失、反向和优化器更新 |

我的学习顺序是先完成一次前向，再学习计时。

**为什么不立即写完整 benchmark：**先确认被测工作能够执行，避免把模型调用问题、设备问题和计时问题同时混在一起。

当前前向的测量目标是：输入已经准备在 GPU 上，模型执行前向，直到 GPU 完成目标计算。模型创建、输入生成和设备迁移属于此前的准备工作。

## 5. 理解 model 与 x 的关系

- `model` 是处理数据的模型实例，拥有结构和权重。
- `x` 是本次交给模型的一批输入。
- `logits` 是模型产生的输出分数。

创建模型不会自动创建这批输入；模型调用时才把二者联系起来。同一个模型可以处理多批不同输入，不必每换一批输入就重新创建模型。

整个语言模型接收 token 编号，Embedding 在内部把编号变成向量。

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| 输入 token 编号 | `[B, T]` | B 条序列，每条 T 个位置 |
| 内部隐藏表示 | `[B, T, D]` | 每个位置用 D 个数字表示 |
| 输出 logits | `[B, T, V]` | 每个位置对 V 个候选 token 给出分数 |

logits 是未归一化分数，不是已经选出的 token，也不是概率。在因果语言模型中，各位置的输出用于预测其后的 token。

## 6. 为什么创建模型需要很多配置

模型类描述一类模型的搭建方式，具体实例需要确定尺寸和结构。

本次试运行配置：

| 配置 | 数值 | 含义 |
| --- | ---: | --- |
| vocab_size | 10000 | 词表大小 |
| context_length | 128 | 配置支持的位置范围 |
| d_model | 512 | 隐藏表示宽度 |
| num_layers | 4 | TransformerBlock 数量 |
| num_heads | 8 | 注意力头数 |
| d_ff | 1024 | 前馈网络中间维度 |
| rope_theta | 10000.0 | RoPE 的频率尺度配置 |
| batch_size | 2 | 本次输入包含两条序列 |
| 实际 sequence_length | 16 | 本次每条输入包含 16 个 token |

配置参数决定模型怎么搭建；可训练权重则是模型搭建后产生、由训练调整的数值。二者不能混淆。

`context_length=128` 与实际输入长度 16 不矛盾：支持的范围与本次使用的长度不同。batch size 描述这一批输入的大小，不需要固定为模型结构的一部分。

**范围说明：**这是用于初步检查的小规模配置，不是讲义 Table 1 的正式模型配置。正式实验需要按对应题目调整条件。

## 7. 编写第一版后，发现三个问题

### 7.1 函数参数之间的分隔符

最初随机输入创建部分，在 `size` 与 `dtype` 参数之间遗漏了分隔符。

**原因：**Python 中换行不等于分隔函数参数。语法问题会在文件解析阶段阻止程序运行。

**修改结果：**补齐分隔符，函数调用能够正常解析。

### 7.2 导入的类名不准确

最初使用 `BasicTransformer`，后来改成 `BasicTransformerLM`，但参考实现的准确名称是 **`BasicsTransformerLM`**，其中 `Basics` 带有末尾的 s。

**原因：**Python 按准确名称查找对象，不会根据名字的含义或相似程度猜测。

**修改结果：**导入和创建实例时都使用参考文件实际定义的类名。

### 7.3 把构造参数误认为同名对象属性

最初通过 `model.vocab_size` 获取词表大小，但参考模型没有设置这个直接属性。

| 名称 | 意义 |
| --- | --- |
| 构造函数里的 `vocab_size` | 创建时传入的参数 |
| 对象上的同名属性 | 必须经过保存或其他明确的属性机制才能存在 |
| `model.config` | 参考实现保存构造配置的位置 |

**原因：**构造函数收到一个参数，不会自动创建同名属性。参数可以用于建立某个层，也可以保存在配置字典中。

**本次修改：**在随机编号上界和输出形状检查中，使用与模型配置一致的词表大小 10000。

**复习提醒：**目前这些数值保持一致；将来改变词表大小时，需要留意相关位置是否同步。此处记录的是本次修改，不代表所有项目都应重复填写同一数值。

## 8. 第一次前向运行成功

在 Windows 项目根目录运行模块入口后，终端输出：

```text
Input shape: torch.Size([2, 16])
Logits shape: torch.Size([2, 16, 10000])
```

程序正常结束，没有出现输出形状断言错误。

**这证明了什么：**在这次环境和配置下，模型能够接受输入、完成前向，并产生预期形状的输出。

**还没有证明什么：**没有确认 GPU 执行、没有验证所有数值性质、没有测量性能，也没有检查反向和优化器。

## 9. 确认显卡与实际执行设备

我使用的显卡是 NVIDIA RTX 3060 Ti。随后检查了 CUDA 可用性、模型第一个参数的设备和输入设备。

第一次检查结果：

```text
if cuda is available: True
model parameters device: cpu
input device: cpu
Input shape: torch.Size([2, 16])
Logits shape: torch.Size([2, 16, 10000])
```

**理解这个结果：**PyTorch 可以使用 CUDA，但这次模型和输入仍在 CPU 上。拥有 NVIDIA 显卡、CUDA 可用、实际在 GPU 上执行，是不同层次的事情。

设备检查的含义：

| 检查 | 回答的问题 |
| --- | --- |
| CUDA 是否可用 | 当前 PyTorch 能否使用 CUDA？ |
| 参数张量的 device | 模型参数实际放在哪里？ |
| 输入张量的 device | 输入数据实际放在哪里？ |

检查第一个参数适合作为本次单设备模型的初步确认；它不等于对任意多设备模型的全面检查。

## 10. 将模型与输入迁移到同一张 GPU

之后在前向之前，使用设备迁移接口将模型与输入都放到 `cuda:0`。

这里学到的区别：

- 模型的 `.to()` 会处理其内部注册的参数和缓冲区。
- 张量跨设备迁移时，`.to()` 返回迁移后的张量，需要让后续输入使用这个返回结果。
- 模型迁移不会自动迁移独立创建的输入，因此两者需要分别确认。

`cuda:0` 表示第一张 CUDA GPU，不是显卡型号。

设备打印放在迁移之后，以检查迁移结果。代码检查还指出：CUDA 可用性输出当时位于迁移之后，这能显示状态，但不能在不可用时提前阻止迁移失败。后续若增强环境检查，应注意这个边界。

## 11. GPU 前向运行成功：当前已达到的里程碑

修改后的实际终端输出：

```text
if cuda is available: True
model parameters device: cuda:0
input device: cuda:0
Input shape: torch.Size([2, 16])
Logits shape: torch.Size([2, 16, 10000])
```

程序正常结束。

**已确认：**

- 参考模型能够被当前 Windows 环境导入和实例化。
- 当前 PyTorch 可以使用 CUDA。
- 检查到的模型参数和输入均位于 `cuda:0`。
- 一次 GPU 前向调用正常返回，输出形状符合预期。
- 迁移设备没有改变输入和输出的逻辑形状。

**尚未完成或未记录验证结果：**

- 输出数值是否全部有限。
- GPU 前向耗时及正确的计时边界。
- 预热与多次测量。
- 均值和标准差统计。
- 损失、反向传播与优化器步骤。
- 按讲义正式模型配置完成实验。

当前默认仍开启梯度追踪。仅前向执行不自动等于关闭求导记录的推理，后续比较时需要说明这一条件。

## 12. 下一步学习：理解计时边界

目前已经知道要测量的工作是什么，但还没有加入计时。

接下来涉及两个工具的概念：

| 工具 | 作用 |
| --- | --- |
| `timeit.default_timer` | 提供计时读数，读数差表示经过的时间 |
| `torch.cuda.synchronize` | 等待相关设备上已提交的 GPU 工作完成 |

接下来需要回答：

1. 开始计时时，是否仍有之前提交的 GPU 工作未完成？
2. 停止计时时，本次前向的 GPU 工作是否真正完成？
3. 记录的是首次执行成本，还是预热后的重复执行成本？

本次学习停在这个思考题：**如果前向调用刚返回就停止计时，而 GPU 仍在计算，记录的时间会漏掉什么？**

## 13. 复习检查表

- [x] 能解释为什么需要 benchmark。
- [x] 能区分模型定义、实例和输入。
- [x] 理解配置参数与模型权重的区别。
- [x] 理解构造参数不会自动成为同名属性。
- [x] 能解释输入 `[B, T]` 与输出 `[B, T, V]`。
- [x] 完成一次前向并通过形状检查。
- [x] 区分 CUDA 可用与实际 CUDA 执行。
- [x] 将模型与输入放到同一 CUDA 设备并完成前向。
- [ ] 验证输出数值是否有限。
- [ ] 理解并建立 GPU 计时边界。
- [ ] 完成预热、重复计时与统计。

今后每完成一步，可以继续记录：**本次目标、修改理由、实际输出、证实了什么、还没有证实什么。**

## 14. 当前代码快照：GPU 前向，准备学习计时

记录日期：2026-09-19。以下内容直接复制自本次读取到的 `cs336_systems/new_benchmark.py`，保留原代码，不补写计时实现。

```python
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=512,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)

batch_size = 2
sequence_length = 16
x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long
)

model = model.to("cuda:0")
x = x.to("cuda:0")


print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)



logits = model(x)

print("Input shape:", x.shape)
print("Logits shape:", logits.shape)
assert logits.shape == (batch_size, sequence_length, 10000)
```

### 本次代码处于哪个阶段

- 已包含参考模型导入、初始化、随机整数输入、模型与输入的 CUDA 迁移、一次前向及形状检查。
- 相比前一次展示的版本，当前文件已导入 `timeit`，不再打印 CUDA 可用性。
- **导入计时模块不等于已经计时。** 当前尚无时钟读数、显式 CUDA 同步、耗时计算、预热或重复测量。
- 前面记录的 GPU 前向成功输出来自此前用户提供的 Windows 运行结果；本次只读取并归档源码，没有重新运行这个快照。

### 下一步为什么研究计时边界

CPU 返回控制权时，GPU 可能仍在计算。计时结束前要能确认目标工作完成；计时开始时还需要分清此前的数据迁移等工作是否结束。模型初始化、输入生成和设备迁移属于准备阶段，当前目标是测量一次前向的执行与完成。

当前练习：理解 `timeit.default_timer` 与 `torch.cuda.synchronize` 各自的职责，然后自行尝试添加计时，再检查测量边界。首次测量用于理解流程，正式性能结论还需要预热和重复测量。

## 15. 单次 GPU 前向计时：代码快照与逐步解释

本节最初按静态阅读检查；随后用户提供了 Windows GPU 运行结果，已记录在第 16 节。本次重新读取源文件，确认下方快照仍与当前代码一致。没有修改或代为运行 Python 文件。

### 修改与原因

- 在模型创建和迁移前检查 CUDA 可用性，不可用时明确终止；补充 GPU 名称输出。
- 在开始读时钟前同步设备，等待此前提交的工作完成，避免把准备阶段的未完成工作混进来。
- 前向调用后先同步，再读取结束时钟，确保测量包含 GPU 完成本次工作所需的等待。
- 秒数乘以 1000 转为毫秒；设备与形状打印位于计时区间之外。
- 本次 `d_model` 从 512 改为 256，属于模型配置变化；不能将不同配置的耗时差异直接归因于计时方法。

### 当前测量的含义

同步边界符合单次 GPU 前向的主机端经过时间测量。该时间包括 CPU 发起计算与等待 GPU 完成的成本，并非各 GPU kernel 耗时的简单合计。计时前同步不会预热模型；当前仍测量首次模型前向，可能包含首次执行准备成本。梯度追踪仍然开启。

尚未完成预热、重复测量、均值和标准差，也未验证输出数值全部有限。实际运行得到的首次前向耗时见第 16 节。

### 本次代码快照

```python
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

# check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)

# create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long
)


# move model and input to cuda
model = model.to("cuda:0")
x = x.to("cuda:0")

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)

# waiting for GPU to be ready
torch.cuda.synchronize(device)

# start timing
start_time = timeit.default_timer()

# operate the forward pass
logits = model(x)

# waiting for GPU to finish
torch.cuda.synchronize(device)

# stop timing
end_time = timeit.default_timer()

# calculate the time taken
time_taken = (end_time - start_time) * 1000 # in milliseconds

print("output device:", logits.device)
print("output shape:", logits.shape)
print(f"single forward pass time: {time_taken:.6f} milliseconds")

assert logits.shape == (batch_size, sequence_length, 10000)
```

### 按代码顺序复习：每一步做什么，为什么做

以下行号对应本节快照以及本次检查时的源文件；今后编辑源文件后，行号可能变化。

| 步骤 | 对应行 | 做了什么 | 为什么这么做 |
| --- | --- | --- | --- |
| 1 | 1–3 | 导入 PyTorch、参考模型类与 timeit | 分别提供张量与设备操作、模型定义、计时工具；导入本身不等于模型已执行或已计时 |
| 2 | 5–9 | 打印 CUDA 可用性；不可用时终止 | 在迁移之前识别环境问题，避免把缺少 CUDA 支持误认为模型计算错误 |
| 3 | 11–12 | 指定第一张 CUDA GPU，并打印名称 | 明确目标设备，记录这次使用 RTX 3060 Ti 的实验背景 |
| 4 | 14–23 | 根据配置创建模型 | 确定被测对象的结构和随机初始化参数；创建成本位于计时之外 |
| 5 | 25–34 | 生成形状为 `[2, 16]` 的整数 token 编号 | 为模型提供符合输入接口的一批数据；编号范围与词表大小一致，生成成本不计入前向 |
| 6 | 37–39 | 将模型与输入分别迁移到 `cuda:0` | 模型与独立创建的输入不会自动一起迁移，计算时需要设备兼容 |
| 7 | 41–43 | 打印参数设备、输入设备和输入形状 | 先检查真正的计算位置和工作负载；把终端打印放在计时区间之外 |
| 8 | 45–46 | 开始计时前同步 | 等待此前提交的 GPU 工作完成，明确目标工作的起始边界；同步不是预热 |
| 9 | 48–49 | 保存开始时钟读数 | 建立测量起点；单个读数本身不是耗时 |
| 10 | 51–52 | 调用模型，得到 logits | 执行当前唯一的被测工作：前向传播；没有反向或优化器更新 |
| 11 | 54–55 | 前向调用后同步 | CPU 调用返回不保证 GPU 完成；等待目标计算结束，避免过早停表 |
| 12 | 57–58 | 保存结束时钟读数 | 在目标 GPU 工作完成后建立测量终点 |
| 13 | 60–61 | 两次读数相减，再乘以 1000 | 时间差得到秒数，转换为更适合阅读的毫秒 |
| 14 | 63–65 | 打印输出设备、输出形状和时间 | 将运行情况与测量结果展示出来；这些打印不计入已结束的测量 |
| 15 | 67 | 检查输出形状 | 验证输出接口符合 `[B, T, V]`，但形状正确不等于数值已全面验证 |

### 计时范围为什么这样划分

- **计时之前：**环境检查、模型初始化、输入生成、设备迁移、输入信息打印以及起始同步。
- **计时之内：**发起模型前向、相关主机端执行与 GPU 工作，以及等待 GPU 完成的末尾同步。
- **计时之后：**计算时间差、打印结果与形状断言。

这测量的是指定前向工作的主机端经过时间，不是整个脚本启动到退出的时间，也不是纯粹的 GPU kernel 时间合计。

起始同步不在计时内；末尾同步在计时内。二者分别帮助排除先前工作、覆盖尚未完成的目标工作。

## 16. 首次 GPU 前向计时运行结果

用户在 Windows 环境运行后提供了以下输出：

```text
if cuda is available: True
GPU device: NVIDIA GeForce RTX 3060 Ti
model parameters device: cuda:0
input device: cuda:0
input shape: torch.Size([2, 16])
output device: cuda:0
output shape: torch.Size([2, 16, 10000])
single forward pass time: 149.476100 milliseconds
```

这验证了本次版本在 RTX 3060 Ti 上完成一次前向及计时，输出设备和形状符合预期。记录的 149.476100 ms 是未预热情况下、带梯度追踪的一次前向的主机端经过时间，计时包含等待 GPU 完成目标工作。

这个数字不能直接代表模型稳定运行时每次前向的成本。首次执行可能包含计算库或执行路径的初始化等成本；仅凭这一个时间值，不能确定各部分的具体贡献，也不能断言性能好坏。时间输出保留六位小数也不等于测量具有相同级别的准确度。

下一阶段学习目标：区分预热执行与正式测量。预热需要实际执行相同的目标工作；同步只负责等待，不替代预热。讲义第 2.1.3(b) 要求 5 次预热、10 次正式测量并报告均值和标准差。目前尚未完成这些步骤，输出数值有限性也尚未记录检查结果。

### 如何解读这次结果

1. **设备符合目标：**模型参数、输入与输出都显示 `cuda:0`，并记录到显卡名称。
2. **形状符合目标：**两条序列、每条 16 个位置，每个位置有 10000 个候选分数。
3. **计时流程已跑通：**在当前代码的同步边界下，得到约 149.48 ms 的单次记录。
4. **尚不能代表稳定性能：**没有 warm-up，也没有多次测量，不能得出稳定均值、波动程度或优化加速比。
5. **不能从数字推断具体开销来源：**首次执行可能有额外准备成本，但本次没有 profile 证据区分各类成本。

### 下一步：为什么需要 warm-up

预热类似正式运动前的热身：先实际执行目标工作，使测量不再只反映第一次使用该执行路径的情况。预热不是等待一段时间，也不是单独调用同步函数。

| 阶段 | 执行的工作 | 耗时是否纳入正式统计 |
| --- | --- | --- |
| 预热 | 与待测模式一致的目标计算 | 否 |
| 正式测量 | 重复执行目标计算并记录时间 | 是 |

预热与正式测量应保持工作负载和执行模式一致，否则预热的不一定是正式测量的路径。讲义要求的 5 次预热、10 次测量是后续正式实验目标，目前未实现，也没有相应结果。

留给自己的思考题：**如果每次正式测量前都重新创建模型，它还是在测同一个模型的重复执行性能吗？** 复习时重点区分：首次执行、同一模型重复执行、模型创建成本这三件事。

### 本阶段进度更新

- [x] 将 CUDA 可用性检查放到设备迁移之前。
- [x] 在 GPU 前向的计时起点和终点处理同步。
- [x] 在 RTX 3060 Ti 上获得一次真实的毫秒计时结果。
- [x] 保存本次源码快照、逐步解释与终端输出。
- [ ] 检查输出数值全部有限。
- [ ] 加入预热并确认正式测量不统计预热耗时。
- [ ] 重复测量，收集均值与标准差。

第 13 节保留了当时的历史进度，本节表示完成单次计时后的最新进度。Python 源文件保持不变。

## 17. 加入 5 次预热与 10 次正式测量：静态检查记录

本次已读取当前代码，尚未收到此版本的 Windows 运行输出。上一阶段的 149.476100 ms 不能当作本版预热后的测量结果。

### 每一步做什么与原因

| 部分 | 做了什么 | 为什么这样做／检查结论 |
| --- | --- | --- |
| 参数 | 设置 5 次预热、10 次正式测量 | 次数与讲义 2.1.3(b) 一致；当前仍使用自选小模型，不是 Table 1 的完整正式实验 |
| 预热 | 在模型和输入迁移完成后重复执行前向 | 预热目标执行路径，而非只等待；使用与正式测量相同的模型与输入 |
| 预热观察 | 每次同步、计时并打印预热耗时 | 便于学习首次与后续执行的区别；预热时间未加入正式时间列表，因此不污染正式统计 |
| 形状检查 | 在第一次预热计时结束后检查输出 | 检查逻辑不在前向计时区间内；仍未检查所有输出数值是否有限 |
| 正式测量 | 每次在开始时钟前同步，在前向后同步再停表 | 确保起点不包含未完成的先前工作，终点覆盖目标 GPU 工作 |
| 结果收集 | 只将正式测量时间加入 time_ms | 使均值与标准差仅基于正式样本 |
| 输出释放 | 每轮计时结束后删除 logits 引用 | 在本脚本中避免将各轮输出及其计算图持续保留；不代表缓存显存一定归还给驱动 |
| 统计 | 计算平均值与除以 N 的标准差 | 当前计算的是总体标准差口径；报告时应说明统计口径 |
| 置信区间输出 | 计算 1.96 乘标准差除以样本数平方根 | 当前打印的是近似区间的半宽，而不是完整上下界；10 次连续测量是否满足相应统计假设尚未验证，不宜无条件称为严格的 95% 置信区间；讲义此处只要求均值和标准差 |

### 当前结论与下一步

从源码看，预热位置、正式测量边界和预热数据排除均符合当前练习目标。带梯度追踪的前向模式仍未改变；当前仅前向，不包含反向与优化器。下一步在同一 Windows CUDA 环境运行，提供 5 次预热、10 次正式测量及统计输出，再判断波动情况。源码未由助手修改或执行。

### 当前代码快照

```python
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long
)


# 4. move model and input to cuda
model = model.to("cuda:0")
x = x.to("cuda:0")

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time: {average_time:.6f} milliseconds")
# calculate the standard deviation
std_dev = (sum((x - average_time) ** 2 for x in time_ms) / len(time_ms)) ** 0.5
print(f"standard deviation: {std_dev:.6f} milliseconds")
# calculate the 95% confidence interval
confidence_interval = 1.96 * std_dev / (len(time_ms) ** 0.5)
print(f"95% confidence interval: {confidence_interval:.6f} milliseconds")









```

## 18. 预热与重复测量成功：实际数据与解释

用户提供的 Windows 运行环境：NVIDIA GeForce RTX 3060 Ti；模型参数、输入、输出均在 cuda:0；输入形状 [2, 16]，输出形状 [2, 16, 10000]；model.training=True，梯度追踪开启。不是梯度数值正确性检查，也没有执行反向。

### 实测数据（毫秒）

| 轮次 | 预热 | 正式测量 |
| --- | ---: | ---: |
| 1 | 149.836600 | 4.411600 |
| 2 | 5.678400 | 5.061700 |
| 3 | 5.498700 | 4.974500 |
| 4 | 4.250600 | 5.056900 |
| 5 | 4.337100 | 5.092600 |
| 6 | — | 4.999200 |
| 7 | — | 5.010400 |
| 8 | — | 4.851900 |
| 9 | — | 4.817600 |
| 10 | — | 5.196600 |

已根据用户提供的 10 个正式样本独立复核：均值 4.947300 ms，总体标准差约 0.207045 ms，最小值 4.411600 ms，最大值 5.196600 ms；标准差／均值约为 4.19%。

### 从结果能理解什么

- 首次预热约 149.84 ms，后续执行约 4–6 ms，明显体现首次执行与重复执行的差异。首次预热约为正式均值的 30.3 倍；这不是模型优化获得了 30 倍加速，也不能把差额全部归因于某一种初始化活动。
- 预热耗时不进入 time_ms，因此正式统计仅代表后 10 次测量。
- 正式耗时在 4.41–5.20 ms 范围内，存在约 4.2% 的相对标准差；没有首次执行那样的极端差异，但不能据此认定完全稳定或定位波动原因。
- 5 次预热不保证耗时单调下降，也不保证消除所有波动。
- 此数据只对应本次自选小模型配置与开启梯度追踪的前向；不能当作讲义 Table 1 small 的结果或整个训练步耗时。

### 这次代码调整及理由

- 模型迁移、输入迁移和同步统一使用 device：减少修改设备时漏改某处的风险。
- 显式设置训练模式，打印 model.training 与梯度开关：让测量口径可见；二者是独立概念。
- 保留每轮计时前后同步、5 次预热和 10 次正式测量：明确边界并区分首次成本与重复执行。
- 移除近似置信区间输出，报告均值、总体标准差、样本数和极值：先报告已理解且可核对的指标。
- 标准差表达式内部的 x 属于生成器表达式局部作用域，不覆盖外部输入张量；命名改善是可读性建议，不是修复覆盖错误。

### 当前代码快照

```python
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient check:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass: {average_time:.6f} milliseconds")
variance = sum((x - average_time) ** 2 for x in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass: {max(time_ms):.6f} milliseconds")








```

### 最新进度

已完成小配置下前向的预热、重复 GPU 计时和基础统计，并拿到用户实际运行结果；助手只核对源码和数据，未在本机执行训练代码。输出有限性、讲义正式配置、反向与优化器测量仍未完成。下一步可以先理解损失如何把 logits 与目标 token 关联起来，为反向传播建立概念基础；不同时改变模型规模与执行范围。

## 19. 创建随机目标张量：静态检查通过

当前增加 targets，用于后续学习损失。源码检查确认其形状为 [2, 16]，包含 32 个编号；dtype 为 torch.long；随机范围为 0（含）至 10000（不含）；直接在指定 CUDA 设备创建。尚未收到本版本实际运行反馈。

后续运行验证已完成，见第 20 节；本节保留最初检查时的状态与代码快照。

### 做了什么，为什么做

- 生成与输入位置对应的目标编号：每个位置提供一个类别标签，而非 10000 个预测分数。
- 打印形状、设备、类型及极值：检查接口是否符合预期；随机样本不必恰好出现 0 或 9999。
- 断言形状、设备、类型和编号范围：将要求转成可执行检查；第一次预热还核对目标与 logits 的设备相同。
- 目标创建与检查位于前向计时之前：不把目标准备成本混入前向时间。读取 GPU 数值的 item 操作可能需要等待设备，当前它们位于计时区间外。
- 标准差表达式使用 t 作为局部变量：命名更清楚；原生成器表达式的 x 本来也不会覆盖输入。

当前仍只测量前向，targets 尚未参与损失计算，也没有反向传播。下一步先运行，确认目标检查通过，再学习预测张量与目标张量如何匹配损失函数的接口。源码未由助手修改或运行。

### 当前代码快照

```python
import torch
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient check:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass: {max(time_ms):.6f} milliseconds")








```

## 20. 目标张量运行验证通过

用户提供的实际结果：RTX 3060 Ti，CUDA 可用；targets 形状 [2, 16]，设备 cuda:0，dtype 为 torch.int64（即 torch.long），最小编号 1400，最大编号 9726；模型参数、输入、输出均在 cuda:0；logits 形状 [2, 16, 10000]。程序正常结束，相关断言通过。模型训练模式与梯度追踪均开启，但未执行反向。

### 本次耗时（毫秒）

| 轮次 | 预热 | 正式测量 |
| --- | ---: | ---: |
| 1 | 143.727700 | 4.569700 |
| 2 | 4.636400 | 4.528100 |
| 3 | 4.762100 | 4.483300 |
| 4 | 4.517600 | 4.626000 |
| 5 | 4.498300 | 4.511500 |
| 6 | — | 4.535300 |
| 7 | — | 4.464300 |
| 8 | — | 4.593100 |
| 9 | — | 4.512800 |
| 10 | — | 4.450000 |

已用正式样本复核：均值 4.527410 ms，总体标准差约 0.053173 ms，范围 4.450000–4.626000 ms，相对标准差约 1.17%。本次组内波动比第 18 节小，但不能仅据两次运行确定原因，不能推断生成 targets 带来了模型加速。目标张量仍未参与损失，测量范围没有扩展为训练步。

### 学到什么与下一步

- long 与 int64 在这里是同一种整数类型，打印名称不同不是错误。
- 32 个随机标签不必覆盖词表两端；1400–9726 满足合法范围。
- 当前已经具备预测分数和目标编号这两类数据。下一步学习损失的含义与接口：每个位置用整组预测分数评价对应目标，批量损失再按选定方式汇总。
- 先理解损失如何关联预测与目标，再添加相关工作；不要把没有损失或反向的本次耗时称为完整训练时间。

代码快照沿用第 19 节；本次只记录用户提供的运行反馈，没有修改或执行源文件。

## 21. 接入一次交叉熵验证：静态检查通过

本次读取用户代码，未运行，也尚无本版损失输出。

### 做了什么与原因

| 位置 | 操作 | 解释 |
| --- | --- | --- |
| 第 138 行 | 正式统计之后重新执行前向 | 此前循环已删除 logits，本次得到新的有效输出与计算图；不改变前面的前向计时范围 |
| 第 141–142 行 | 检查预测的位置维与目标形状、设备一致 | 确认每个位置有对应目标，且设备兼容 |
| 第 145–146 行 | logits 合并前两个维度，targets 同序展开 | 从 [2,16,10000] 与 [2,16] 变为 [32,10000] 与 [32]，保留位置对应关系 |
| 第 152–156 行 | 将原始 logits 和整数目标交给 PyTorch 交叉熵，取 mean | 未提前 softmax；当前无权重、无忽略位置、无标签平滑，得到 32 个位置的平均损失 |
| 第 159–168 行 | 打印并断言标量形状、有限性、设备及 requires_grad | 验证损失接口和求导准备状态；requires_grad=True 不等于已经执行反向或验证所有参数梯度 |
| 第 171 行 | 删除本次验证张量引用 | 当前仅验证损失，结束后释放相关引用合理；未来学习反向时需重新考虑损失和计算图的生命周期 |

item 与有限性检查在前向计时完成之后，不计入已有样本。当前新增的是功能验证，尚未测量损失耗时、反向或优化器。本次静态检查未发现损失接口、形状组织或计时范围方面的明显问题。下一步在 Windows GPU 环境运行并提供新增损失部分的输出。

### 当前代码快照

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient check:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass: {max(time_ms):.6f} milliseconds")

# 9. calculate the loss
print("\n calculating the loss...")

# re-operate the forward pass
logits = model(x)

# validate the output shape
assert logits.shape[:2] == targets.shape
assert logits.device == targets.device

# merge the batch dimension and sequence length dimension
flat_logits = logits.reshape(-1, logits.shape[-1])
flat_targets = targets.reshape(-1)

print("flattened logits shape:", flat_logits.shape)
print("flattened targets shape:", flat_targets.shape)

# calculate the loss
loss = F.cross_entropy(
    flat_logits,
    flat_targets,
    reduction="mean",
)

# validate the loss
print("loss:", loss.item())
print("loss shape:", loss.shape)
print("loss device:", loss.device)
print("loss requires_grad:", loss.requires_grad)
print("loss is finite:", torch.isfinite(loss).item())

assert loss.shape == torch.Size([])
assert torch.isfinite(loss).item()
assert loss.device == x.device
assert loss.requires_grad

# 10 delete relevant tensors
del logits, flat_logits,flat_targets, loss





```

## 22. 损失计算实际运行验证通过

用户在 RTX 3060 Ti 的 Windows 环境运行第 21 节版本后提供结果。目标形状 [2,16]，dtype int64，设备 cuda:0，编号范围 652–9923；模型训练模式和梯度追踪开启。

### 本次时间记录（毫秒）

预热：145.525000、4.863500、6.005500、4.933700、5.026300。

正式测量：4.823100、4.863100、4.878800、4.872500、4.877800、4.969000、5.075600、4.837000、4.934600、4.871100。

正式均值 4.900260 ms，总体标准差约 0.071113 ms，最小值 4.823100 ms，最大值 5.075600 ms。已由用户提供样本复核。范围仍是仅前向，不包括后面的损失验证。

### 损失输出

```text
flattened logits shape: torch.Size([32, 10000])
flattened targets shape: torch.Size([32])
loss: 9.226762771606445
loss shape: torch.Size([])
loss device: cuda:0
loss requires_grad: True
loss is finite: True
```

### 结果说明

- 预测与目标按 32 个分类位置组织，接口和形状检查通过。
- mean 汇总产生标量张量；空形状 [] 不是没有数据，而是一个没有轴的标量。
- 损失在 GPU 上，是有限值，且可以参与自动求导。
- 对 10000 类完全均匀预测，单位置交叉熵为 ln(10000)，约 9.21034。本次 9.22676 与之接近，与未训练模型和随机目标的情境相容，但不是正确性的充分证明；随机初始化不保证均匀预测，接近该值也不能证明所有位置分布均匀。
- 尚未调用反向或更新参数，requires_grad=True 不等于参数已经有梯度；当前损失不是衡量真实语言能力的结果。

下一阶段学习目标：理解反向传播产生的梯度与优化器更新的区别，再验证一次反向。当前源码末尾会删除损失及相关张量，后续应自行考虑反向需要它们存活到何时。尚未记录反向实现或运行结果。代码快照沿用第 21 节，本次没有修改或运行源码。

## 23. 一次反向传播验证：静态检查通过，等待运行结果

本次只读检查用户新增代码，没有代为运行或修改 Python 文件。

### 每一步做什么与原因

| 位置 | 操作 | 为什么这样做 |
| --- | --- | --- |
| 第 175 行 | 清理参数梯度，设置为 None | 明确本次验证的初始状态，避免旧梯度干扰；放在前向之后、反向之前也合理，不会切断已建立的计算图 |
| 第 177 行 | 选取 lm_head.weight | 输出层参与当前损失计算，适合做单个参数的初步检查；当前形状应为 [10000,256] |
| 第 180–185 行 | 检查参数需要梯度，反向前 grad 为 None | 区分参数可求导与参数已经拥有梯度 |
| 第 188 行 | 在 loss 仍存活时执行一次 backward | 损失保留求导关系，可沿图计算梯度；未创建或调用优化器 |
| 第 190–197 行 | 检查梯度存在、形状与参数一致、所有元素有限 | 验证输出层参数收到格式合理且无 NaN/Inf 的梯度；不能据此证明所有参数梯度正确，也没有检查梯度是否非零 |
| 第 200–202 行 | 再清梯度，删除验证张量引用 | 在检查完成后结束当前验证的生命周期，避免保留不再需要的数据 |

反向位于前向计时和统计之后，未计入已有前向样本。此处没有单独显式同步并非计时问题，因为没有测反向耗时；同一默认执行流中的后续梯度运算遵循执行顺序，有限性检查的 item 读取会等待相应结果。当前源码逻辑可进入实际运行验证，但尚无本版运行证据。

预期观察：参数 requires_grad=True，参数形状 [10000,256]，反向前 grad=None，反向后梯度存在、形状 [10000,256] 且有限。仅检查一个参数，不等于整个反向数值正确性测试。

### 当前代码快照

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient check:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass: {max(time_ms):.6f} milliseconds")

# 9. calculate the loss
print("\n calculating the loss...")

# re-operate the forward pass
logits = model(x)

# validate the output shape
assert logits.shape[:2] == targets.shape
assert logits.device == targets.device

# merge the batch dimension and sequence length dimension
flat_logits = logits.reshape(-1, logits.shape[-1])
flat_targets = targets.reshape(-1)

print("flattened logits shape:", flat_logits.shape)
print("flattened targets shape:", flat_targets.shape)

# calculate the loss
loss = F.cross_entropy(
    flat_logits,
    flat_targets,
    reduction="mean",
)

# validate the loss
print("loss:", loss.item())
print("loss shape:", loss.shape)
print("loss device:", loss.device)
print("loss requires_grad:", loss.requires_grad)
print("loss is finite:", torch.isfinite(loss).item())

assert loss.shape == torch.Size([])
assert torch.isfinite(loss).item()
assert loss.device == x.device
assert loss.requires_grad

# 10. validate backward pass without timing

print("\n validating backward pass without timing...")

# clear the gradients
model.zero_grad(set_to_none=True)
# select the model parameters
param = model.lm_head.weight

print("\n validate backward pass...")
print("if parameters needed gradient:", param.requires_grad)
print("parameter shape:", param.shape)
print("graditent before backward pass:", param.grad)

assert param.requires_grad
assert param.grad is None

# start backward from loss
loss.backward()
# validate the gradients
print("check if gradient exists after backward pass:", param.grad is not None)
assert param.grad is not None

print("gradient shape:", param.grad.shape)
assert param.grad.shape == param.shape

print("check if gradient is finite:", torch.isfinite(param.grad).all().item())
assert torch.isfinite(param.grad).all().item()

# clear the gradients after validationalidation finish
model.zero_grad(set_to_none=True)
# delete relevant tensors
del logits, flat_logits,flat_targets, loss, param
```

## 24. 一次反向传播实际验证通过

用户在 Windows / RTX 3060 Ti 上运行并提供结果。梯度追踪标签已改为 gradient tracking enabled，避免误认为梯度正确性检查。

### 功能验证结果

- targets：[2,16]、int64、cuda:0，最小编号 1000，最大编号 9523。
- logits 展平为 [32,10000]，targets 展平为 [32]。
- loss=9.264738082885742，形状 []，设备 cuda:0，requires_grad=True，数值有限。
- 所选 lm_head.weight：requires_grad=True，形状 [10000,256]。
- 清理后、反向前，grad=None；执行反向后，grad 存在，形状 [10000,256]，全部有限。

这表明本次损失能够反向到选定的输出层权重，梯度存在性、形状和有限性检查通过；不等于验证了所有参数梯度或进行了数值梯度校验。没有执行优化器更新，也没有单独实测参数前后相等。

### 本次前向计时（毫秒）

预热：159.226200、5.040300、4.774900、4.782100、4.782600。

正式测量：4.577500、4.498600、6.025800、6.180800、6.189300、4.734400、4.657100、5.033900、4.750700、4.702800。

已复核正式均值 5.135090 ms，总体标准差约 0.666945 ms，最小值 4.498600 ms，最大值 6.189300 ms；相对标准差约 12.99%。第 3–5 次约 6 ms，其余多数约 4.5–5.0 ms，波动比前一次明显。不能仅据这组数字定位原因，也不应无依据删除较慢样本。损失与反向在计时完成后执行，不能把该均值称为前向加反向耗时。

### 当前进度与下一个概念

已完成小配置的前向计时、损失检查、单参数反向功能验证。下一步学习重复执行前向与反向时的两个边界：每轮参数梯度是否应独立，以及本轮反向对应的是哪一次前向产生的计算图。默认反向后通常会释放所保存的中间数据，不能把同一个旧 loss 当作可无限重复反向的独立实验。预热模式与正式测量模式也应一致。具体实现留待用户尝试后检查。

本节记录用户终端反馈；源码主体快照见第 23 节，随后输出标签的修改与本次反馈一致。助手没有运行 GPU 程序。

## 25. 两轮前向与反向验证：静态检查通过

日期：2026-09-20。当前只读取源码，没有运行；等待用户 Windows GPU 反馈。

### 步骤与理由

| 行号 | 本次实现 | 检查结论 |
| --- | --- | --- |
| 138–139 | 循环外选取输出层权重并确认可求导 | 两轮观察同一参数对象，清梯度不会替换权重对象 |
| 145–148 | 每轮清理梯度并检查 grad 为 None | 为每轮建立独立梯度初始状态，避免累积上一轮结果 |
| 151–169 | 每轮重新前向、整理形状并计算损失 | 每轮使用新的求导图和损失，没有复用已反向的旧 loss；检查标量、有限性、设备和求导标记 |
| 174 | 对本轮 loss 执行反向 | 每张图只反向一次，无需 retain_graph |
| 177–184 | 检查选定参数的梯度存在、形状及有限性 | 满足本次初步功能验证，仍不是全模型梯度数值正确性证明 |
| 187 | 本轮检查后删除输出与损失引用 | 不会在反向之前释放本轮 loss；参数梯度由下一轮开始时清理 |
| 190–191 | 两轮结束后清理最后一轮梯度与观察引用 | 验证结束后不保留最后一轮梯度 |

两轮验证都位于原前向计时统计之后，原有数字仍是仅前向耗时。当前没有优化器步骤；同一模型、输入与目标下，两轮损失可能相同或接近，这不意味着复用了旧图，图是否重新建立应由执行路径判断。没有发现阻止当前两轮验证目标的明显问题，下一步运行并提供两轮输出。

### 当前源码快照

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
    # release the output of the warmup run
    del logits

# 7. change from a single timer to 10 official measurements
print("\n measurement starts...")
time_ms = []
for i in range(measurement_runs):
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"measurement run {i+1} time: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits

# 8. statistics the measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass: {max(time_ms):.6f} milliseconds")

# 9. two forward and backward verification, no timing
print("\n two forward and backward verification, no timing starts...")

# select the output layer weights for observation
param = model.lm_head.weight
assert param.requires_grad

for i in range(2):
    print(f"\n this is the {i + 1} forward and backward verification")

    # ① clear the gradient
    model.zero_grad(set_to_none=True)

    print("the gradient after clearing: ", param.grad)
    assert param.grad is None

    # ② re-execute the forward pass for each round
    logits = model(x)

    assert logits.shape[:2] == targets.shape
    assert logits.device == targets.device

    # ③ reshape the logits and targets, and re-calculate the loss for each round
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert loss.device == x.device
    assert loss.requires_grad

    print("the loss of this round: ", loss.item())

    # ④ perform the backward propagation for the loss of this round
    loss.backward()

    # ⑤ check the gradient
    assert param.grad is not None
    assert param.grad.shape == param.shape
    assert torch.isfinite(param.grad).all().item()

    print("the gradient after backward propagation exists: ", param.grad is not None)
    print("the shape of the parameter: ", param.shape)
    print("the shape of the gradient: ", param.grad.shape)
    print("the gradient finite value check passed")

    # ⑥ after the check of this round, release the output and loss of this round
    del logits, flat_logits, flat_targets, loss

# after the two forward and backward verifications, clear the gradient of the last round
model.zero_grad(set_to_none=True)
del param

print("\n two forward and backward verification, no timing ends...")
```

## 26. 两轮独立前向与反向实际验证通过

记录日期：2026-09-20。用户在 Windows、RTX 3060 Ti 上运行第 25 节版本并提供输出；助手未运行 GPU 程序。

### 环境与目标检查

CUDA 可用，模型参数、输入与输出设备为 cuda:0；输入 [2,16]、输出 [2,16,10000]。targets 为 int64，形状 [2,16]，编号范围 802–9699。模型训练模式、梯度追踪均开启。

### 两轮验证结果

| 检查项 | 第 1 轮 | 第 2 轮 |
| --- | --- | --- |
| 清理后梯度 | None | None |
| 损失 | 9.320413589477539 | 9.320413589477539 |
| 反向后梯度存在 | True | True |
| 参数形状 | [10000,256] | [10000,256] |
| 梯度形状 | [10000,256] | [10000,256] |
| 梯度有限性 | 通过 | 通过 |

这说明两轮均完成初步功能检查。结合源码，每轮都重新前向并计算新损失，没有复用旧计算图；清理后 grad=None，支持两轮没有沿用上一轮梯度。检查范围仍为选定输出层参数，不等于全模型梯度数值正确性验证。

两轮损失相同符合本次没有优化器更新、复用同一模型与输入目标的情境。相同数值不代表相同计算图，也不代表梯度计算无效。脚本没有直接比较权重前后值，不把此现象当作权重未变化的独立证明。

### 前向计时记录（毫秒）

预热：152.743300、5.472000、5.382900、5.295600、4.724100。

正式样本：4.703600、5.871000、4.772800、5.632100、5.137800、4.850300、4.704600、4.925100、4.808300、4.667500。

已复核：均值 5.007310 ms，总体标准差约 0.397329 ms，最小值 4.667500 ms，最大值 5.871000 ms，相对标准差约 7.93%。它仍是仅前向的测量结果；后面的两轮反向没有计时。不能根据这些数据计算反向耗时。

### 下一步学习目标

当前已经验证可重复执行前向、损失和反向。下一步先定义新的测量范围：只测反向，还是测前向、损失与反向的合计；二者不同。原来的前向预热不等于对反向也完成预热，正式测量与预热的工作模式应一致。打印、形状和有限性验证属于功能检查，其位置需要与目标计时范围区分。清梯度是否计入时间也应明确记录。具体实现由用户尝试后再检查。

本阶段最新进度：两轮反向功能验证已通过；前向加反向计时尚未实现或验证，优化器更新也尚未加入。源码快照见第 25 节。

## 27. 扩展前向、损失、反向合计计时：发现两处检查逻辑问题

日期：2026-09-20。静态阅读，未运行或修改源码。

### 已符合约定的部分

预热和正式测量均执行新的前向、交叉熵损失与反向；每轮清梯度在起始同步与开始读时钟之前，排除在计时之外；反向后同步再停止计时；打印、断言与删除输出引用均在计时之外。正式样本仍为 10 次，不包含 5 次预热。统计标签已更新为前向加损失加反向，没有优化器更新。

### 需要自行修正的两处检查

- 第 123 行位于第 97 行 backward 之后，却要求输出层参数的梯度为 None。这与正常反向后的预期状态矛盾；正常执行时会在第一次预热检查处触发断言，无法到达正式测量。应对照文件后部的两轮验证，区分清理后与反向后的检查条件。
- 第 124 行把参数形状与其自身比较，无法验证梯度形状。应思考此处真正需要比较的两个对象。
- 第 125 行依赖梯度是张量，不能在没有确认梯度存在的情况下进行有限性检查。

### 其余说明

末尾两轮不计时验证仍保留，属于额外功能检查，不影响前面计时样本，当前不必为此重构。此版本未运行，不记录合计耗时；先修改检查逻辑，再运行验证。

### 本次待修正代码快照

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)

    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)

    #=======================
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )
    loss.backward()

    #=======================



    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss).item()
        assert loss.device == x.device
        assert loss.requires_grad

        param = model.lm_head.weight
        assert param.grad is None
        assert param.shape == param.shape
        assert torch.isfinite(param.grad).all().item()
        
        print("the loss of the first warmup run: ", loss.item())
        print("the check of loss and output gradient passed")
        del param
    # release the output of the warmup run
    del logits, flat_logits, flat_targets, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )
    loss.backward()
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"formal measurement run {i+1} time:" 
    f"forward pass + loss + backward pass: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits, flat_logits, flat_targets, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass + loss + backward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass + loss + backward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass + loss + backward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass + loss + backward pass: {max(time_ms):.6f} milliseconds")

# 9. two forward and backward verification, no timing
print("\n two forward and backward verification, no timing starts...")

# select the output layer weights for observation
param = model.lm_head.weight
assert param.requires_grad

for i in range(2):
    print(f"\n this is the {i + 1} forward and backward verification")

    # ① clear the gradient
    model.zero_grad(set_to_none=True)

    print("the gradient after clearing: ", param.grad)
    assert param.grad is None

    # ② re-execute the forward pass for each round
    logits = model(x)

    assert logits.shape[:2] == targets.shape
    assert logits.device == targets.device

    # ③ reshape the logits and targets, and re-calculate the loss for each round
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert loss.device == x.device
    assert loss.requires_grad

    print("the loss of this round: ", loss.item())

    # ④ perform the backward propagation for the loss of this round
    loss.backward()

    # ⑤ check the gradient
    assert param.grad is not None
    assert param.grad.shape == param.shape
    assert torch.isfinite(param.grad).all().item()

    print("the gradient after backward propagation exists: ", param.grad is not None)
    print("the shape of the parameter: ", param.shape)
    print("the shape of the gradient: ", param.grad.shape)
    print("the gradient finite value check passed")

    # ⑥ after the check of this round, release the output and loss of this round
    del logits, flat_logits, flat_targets, loss

# after the two forward and backward verifications, clear the gradient of the last round
model.zero_grad(set_to_none=True)
del param

print("\n two forward and backward verification, no timing ends...")
```

## 28. 合计计时复查：预热后的梯度断言已修正

日期：2026-09-20。本次仅静态检查，未修改或执行源文件。

- 第 122 行已改为反向后确认梯度存在，符合该阶段的预期状态。
- 第 123 行已比较梯度形状与参数形状，替代原先无意义的自身比较。
- 第 124 行在确认梯度存在后检查有限性，依赖顺序合理。
- 清梯度仍位于开始时钟之前；预热与正式测量均包含新前向、损失与反向；结束时钟在 GPU 同步之后；检查和打印位于计时之外。
- 只统计 10 次正式样本，结果标签准确描述合计范围，未包含优化器更新。
- 原先末尾两轮额外验证被三引号包成字符串，不再执行。三引号在 Python 中是字符串语法，不是真正的块注释；当前不构成运行阻塞。

当前没有发现必须修改的明显问题，下一步在 Windows GPU 环境运行并提供预热、首次验证、正式样本与统计输出。静态检查通过不等于已取得本版本合计计时结果。

### 当前代码快照

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)

    # waiting for GPU to be ready
    torch.cuda.synchronize(device)
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)

    #=======================
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )
    loss.backward()

    #=======================



    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(f"warmup run {i+1} time: {time_taken:.6f} milliseconds")

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert targets.device == logits.device  
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss).item()
        assert loss.requires_grad

        param = model.lm_head.weight
        assert param.grad is not None
        assert param.grad.shape == param.shape
        assert torch.isfinite(param.grad).all().item()
        
        print("the loss of the first warmup run: ", loss.item())
        print("the check of loss and output gradient passed")
        del param
    # release the output of the warmup run
    del logits, flat_logits, flat_targets, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits = model(x)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )
    loss.backward()
    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(f"formal measurement run {i+1} time:" 
    f"forward pass + loss + backward pass: {time_taken:.6f} milliseconds")
    # release the output of the measurement run
    del logits, flat_logits, flat_targets, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
print(f"average time of forward pass + loss + backward pass: {average_time:.6f} milliseconds")
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"standard deviation of forward pass + loss + backward pass: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of forward pass + loss + backward pass: {min(time_ms):.6f} milliseconds")
print(f"max time of forward pass + loss + backward pass: {max(time_ms):.6f} milliseconds")


""" # 9. two forward and backward verification, no timing
print("\n two forward and backward verification, no timing starts...")

# select the output layer weights for observation
param = model.lm_head.weight
assert param.requires_grad

for i in range(2):
    print(f"\n this is the {i + 1} forward and backward verification")

    # ① clear the gradient
    model.zero_grad(set_to_none=True)

    print("the gradient after clearing: ", param.grad)
    assert param.grad is None

    # ② re-execute the forward pass for each round
    logits = model(x)

    assert logits.shape[:2] == targets.shape
    assert logits.device == targets.device

    # ③ reshape the logits and targets, and re-calculate the loss for each round
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert loss.device == x.device
    assert loss.requires_grad

    print("the loss of this round: ", loss.item())

    # ④ perform the backward propagation for the loss of this round
    loss.backward()

    # ⑤ check the gradient
    assert param.grad is not None
    assert param.grad.shape == param.shape
    assert torch.isfinite(param.grad).all().item()

    print("the gradient after backward propagation exists: ", param.grad is not None)
    print("the shape of the parameter: ", param.shape)
    print("the shape of the gradient: ", param.grad.shape)
    print("the gradient finite value check passed")

    # ⑥ after the check of this round, release the output and loss of this round
    del logits, flat_logits, flat_targets, loss

# after the two forward and backward verifications, clear the gradient of the last round
model.zero_grad(set_to_none=True)
del param

print("\n two forward and backward verification, no timing ends...") """
```

## 29. 前向、损失、反向合计计时实际运行通过

记录日期：2026-09-20。用户提供 Windows / RTX 3060 Ti 运行输出；源码快照见第 28 节，助手未运行 GPU 程序。

### 运行状态

CUDA 可用；模型参数与输入在 cuda:0，输入 [2,16]，输出 [2,16,10000]；targets 为 int64，形状 [2,16]，范围 281–9994。模型训练模式、梯度追踪开启。第一次预热的损失为 9.222169876098633，输出与所选输出层梯度检查通过。

### 实测数据（毫秒）

| 轮次 | 预热 | 正式测量 |
| --- | ---: | ---: |
| 1 | 258.670600 | 13.154400 |
| 2 | 13.828000 | 11.765400 |
| 3 | 14.365600 | 13.056700 |
| 4 | 11.624200 | 13.069900 |
| 5 | 13.849000 | 11.655200 |
| 6 | — | 12.119700 |
| 7 | — | 14.128100 |
| 8 | — | 11.795000 |
| 9 | — | 11.697100 |
| 10 | — | 11.810200 |

根据正式样本独立复核：均值 12.425170 ms，总体标准差约 0.816891 ms，最小值 11.655200 ms，最大值 14.128100 ms，相对标准差约 6.57%。

### 测量范围与解释

- 时间覆盖新的前向、损失所需的形状整理与交叉熵、反向，以及等待 GPU 完成的同步；清梯度、初始化、输入生成、设备迁移、打印和检查不计入。
- 每轮反向前梯度置为 None；反向内部建立梯度存储的成本仍计入。
- 5 次预热与正式测量工作模式一致；首次预热明显慢于后续，但没有 profiler 证据把差额分配给某个具体阶段。
- 本次是合计时间，不是反向单独耗时；不能把它与历史另一轮前向均值简单相减就当作精确反向时间，差值还混有损失计算、运行差异与噪声。
- 此处没有优化器更新，不能称为完整训练步。结果仅对应当前小配置，不是讲义 Table 1 small 的正式结果。
- 存在约 6.57% 的组内相对标准差，暂不推断波动原因，不无依据删除较慢样本。

### 当前里程碑

已完成仅前向计时，以及独立梯度条件下前向、损失、反向的重复合计计时。下一步可以学习优化器如何消费梯度并更新参数，先验证一次更新，再考虑纳入完整训练步计时。当前尚未实现或验证优化器更新。


## 30. 增加模式开关：用同一脚本测量两条路径

记录日期：2026-09-20。本节记录学生自行完成的实现、代码复查及用户提供的 Windows / RTX 3060 Ti 运行结果。助手未执行 GPU 测试；末尾代码为本地文件原样快照，不是新生成的实现。

### 为什么增加模式开关

此前分别写过只前向与前向、损失、反向的测量流程。手动修改代码切换容易漏改预热、正式测量或标签。模式开关将“测什么”集中为一个选项，并复用模型初始化、输入准备、计时与统计。

| 模式 | 每轮执行内容 | 梯度口径 |
| --- | --- | --- |
| forward | 模型前向 | 训练模式、开启梯度追踪，建立计算图但不执行反向 |
| forward_backward | 模型前向、形状整理、平均交叉熵、反向 | 使用本轮新计算图产生参数梯度 |

只前向不等于关闭梯度追踪的推理。此阶段保留训练前向口径；两种模式均没有优化器更新。

### 每一步做了什么，为什么这样做

1. 使用 argparse 定义 --mode，可选 forward 和 forward_backward，默认 forward。目的是在运行时切换，避免编辑源文件。
2. 定义 measurement_name。预热、逐轮测量与最终统计都使用同一标签，避免结果名称与执行路径不一致。
3. 把每轮工作集中到 run_steps()。两种模式都执行前向，仅反向模式计算损失并 backward；只前向时 loss 为 None。预热与正式测量都调用该函数，确保预热的是正式要测的路径。
4. 每轮先清理梯度，再同步，再开始计时。梯度不会跨轮累加，清理梯度的耗时不算入本次约定的范围。
5. 执行该模式的工作后同步，再停止时钟。CPU 提交完任务不代表 GPU 已完成，需要等待后再读结束时间。
6. 首轮预热在计时后验证输出；只有反向模式验证损失与所选输出层参数梯度。这样不会在只前向模式访问不存在的损失。
7. 每轮结束后释放 logits 和 loss 的引用；打印、检查、释放均在计时之外。
8. 仅统计 10 次正式测量，计算均值、总体标准差（除以 N）、最小和最大值；5 次预热不混入统计。

注意：清梯度在计时外，不代表反向过程中分配梯度存储的开销也被排除。使用主机时钟加 CUDA 同步测得的是该路径的墙钟耗时，包含主机调度等开销，不能解释为纯 GPU kernel 时间。

### 复查时发现并修正的事项

- 初版命令行选项叫 --model，实际含义是模式。学生已统一为 --mode、args.mode 和 Invalid mode。
- 初版逐轮正式输出写死为前向、损失、反向，导致 forward 模式日志误标。学生已让预热和正式输出统一使用 measurement_name。
- 修正后的静态检查没有发现影响当前双模式运行的明显问题；随后用户分别运行两种模式验证。
- 多行打印的缩进可以整理，但括号内部的当前缩进不构成语法错误。
- 只前向时仍创建 targets、调用 zero_grad，属于计时外的多余工作，不是当前必须修改的问题。
- 可选改进：只前向分支补充输出梯度追踪、loss 为 None、参数 grad 为 None 的检查；开头打印模式、模型配置与 dtype；整理末尾不执行的旧验证字符串。这些是建议，不应当作已经完成。
- 所选参数梯度存在、形状正确且有限，只是基本检查，不代表所有参数梯度或梯度数值正确性都已验证。

### 两种模式实际运行

用户在 Windows 虚拟环境中分别运行模块，附加选项 --mode forward 与 --mode forward_backward。

共同状态：GPU 为 NVIDIA GeForce RTX 3060 Ti，模型和输入位于 cuda:0；输入 [2,16]，输出 [2,16,10000]；训练模式和梯度追踪均开启；每个模式 5 次预热、10 次正式测量。

当前玩具配置：vocab_size=10000、context_length=128、d_model=256、num_layers=4、num_heads=8、d_ff=1024、rope_theta=10000，batch_size=2，实际 sequence_length=16。它不是作业表 1 的 small 配置。两次独立启动没有固定随机种子，不能声称使用了完全相同的权重和随机输入。

前向模式 targets 范围 202–9824；反向模式为 298–9551，均是合法 int64 token 编号。反向模式首轮预热检查通过，但本次日志没有打印损失的具体数值。

### 原始耗时（单位 ms）

| 轮次 | 前向预热 | 前向正式测量 | 前向＋损失＋反向预热 | 前向＋损失＋反向正式测量 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 150.621700 | 5.315900 | 248.714800 | 13.422900 |
| 2 | 4.970500 | 4.956300 | 14.457400 | 11.699800 |
| 3 | 5.031300 | 4.928300 | 11.804100 | 11.785100 |
| 4 | 4.781200 | 4.891500 | 13.283900 | 12.536400 |
| 5 | 4.804100 | 6.176900 | 11.973700 | 11.962300 |
| 6 | — | 5.737200 | — | 13.363600 |
| 7 | — | 4.920300 | — | 11.675500 |
| 8 | — | 4.971800 | — | 11.729100 |
| 9 | — | 5.900800 | — | 12.892900 |
| 10 | — | 4.986500 | — | 11.842600 |

| 指标 | 只前向 | 前向＋损失＋反向 |
| --- | ---: | ---: |
| 平均耗时（ms） | 5.278550 | 12.291020 |
| 总体标准差（ms） | 0.457170 | 0.668080 |
| 最小耗时（ms） | 4.891500 | 11.675500 |
| 最大耗时（ms） | 6.176900 | 13.422900 |

### 如何理解这些结果

- 两种模式的预热、正式测量和统计标签都正确切换，均完成运行；模式开关阶段已取得实际验证。
- 首轮预热明显较慢，已排除在统计之外。不能仅凭日志确定具体初始化或 kernel 开销占比。
- 两次测量的均值比约为 2.33，仅描述当前配置和这两次运行，不是所有 Transformer 的固定比例。
- 不能把均值之差直接认作精确的反向耗时，因为还混有损失计算、两次启动差异与测量波动。
- 不因个别轮次较慢就认定代码错误，也不无依据剔除较慢样本。
- forward 输出前出现的一条旧反向日志位于该次启动命令之前，不属于该次运行。
- 尚未测量优化器更新，也未完成全部作业实验。

### 下一步候选：从“测什么”到“测多大”

先考虑把 batch_size 和 sequence_length 开放为参数，并记录在日志中。保持模型和其他条件不变，每次只改变一个因素，观察两种模式耗时随工作量变化。序列长度实验需要留意模型当前 context_length=128 的配置边界，不应直接假定任意长度都可用。

这是下一阶段建议，尚未据此修改代码或开展实验。第 29 节提到的优化器学习仍可作为后续方向，此后实际优先完成的是本节模式开关。

### 学生代码原样快照

以下为补记时读取的 new_benchmark.py，保留学生当前格式及末尾未执行的旧验证字符串。

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", 
    choices = ["forward", "forward_backward"],
    default = "forward",
    help = "forward: forward pass only, forward_backward: forward pass + loss + backward pass"
    )
args = parser.parse_args()
mode = args.mode
if mode == "forward":
    measurement_name = "forward time"
elif mode == "forward_backward":
    measurement_name = "forward pass + loss + backward pass time"
else:
    raise ValueError(f"Invalid mode: {mode}")

# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))

# 2. create model
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    rope_theta=10000.0,
)
model = model.to(device)
model.train()

# 3. create input
batch_size = 2
sequence_length = 16

x = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
)


# 4. move model and input to cuda
x = x.to(device)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=10000, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < 10000)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs
def run_steps():
    logits = model(x)
    loss = None
    if mode == "forward_backward":
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="mean",
        )
        loss.backward()
    return logits, loss

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)


    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits, loss = run_steps()
    #=======================


    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(
    f"warmup run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, 10000)
        assert logits.device == x.device 

        if mode == "forward_backward":
            assert loss is not None
            assert loss.shape == torch.Size([])
            assert loss.requires_grad
            assert torch.isfinite(loss).item()
            assert loss.device == targets.device

            param = model.lm_head.weight
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert torch.isfinite(param.grad).all().item()
            del param 

        print("the present mode check passed")
    # release the output of the warmup run
    del logits, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()

    logits, loss = run_steps()

    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(
    f"formal measurement run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )
    # release the output of the measurement run
    del logits, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"measurement name: {measurement_name}")
print(f"average time of {measurement_name}: {average_time:.6f} milliseconds")
print(f"standard deviation of {measurement_name}: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of {measurement_name}: {min(time_ms):.6f} milliseconds")
print(f"max time of {measurement_name}: {max(time_ms):.6f} milliseconds")


""" # 9. two forward and backward verification, no timing
print("\n two forward and backward verification, no timing starts...")

# select the output layer weights for observation
param = model.lm_head.weight
assert param.requires_grad

for i in range(2):
    print(f"\n this is the {i + 1} forward and backward verification")

    # ① clear the gradient
    model.zero_grad(set_to_none=True)

    print("the gradient after clearing: ", param.grad)
    assert param.grad is None

    # ② re-execute the forward pass for each round
    logits = model(x)

    assert logits.shape[:2] == targets.shape
    assert logits.device == targets.device

    # ③ reshape the logits and targets, and re-calculate the loss for each round
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert loss.device == x.device
    assert loss.requires_grad

    print("the loss of this round: ", loss.item())

    # ④ perform the backward propagation for the loss of this round
    loss.backward()

    # ⑤ check the gradient
    assert param.grad is not None
    assert param.grad.shape == param.shape
    assert torch.isfinite(param.grad).all().item()

    print("the gradient after backward propagation exists: ", param.grad is not None)
    print("the shape of the parameter: ", param.shape)
    print("the shape of the gradient: ", param.grad.shape)
    print("the gradient finite value check passed")

    # ⑥ after the check of this round, release the output and loss of this round
    del logits, flat_logits, flat_targets, loss

# after the two forward and backward verifications, clear the gradient of the last round
model.zero_grad(set_to_none=True)
del param

print("\n two forward and backward verification, no timing ends...") """
```


## 31. 模型配置参数化、预设与覆盖，以及 small 配置实测

记录日期：2026-09-20。代码由学生完成；助手做静态检查，并应用户要求删除重复配置打印。GPU 运行结果来自用户提供的 Windows / RTX 3060 Ti 日志，助手未运行 GPU 实验。

### 这一步要解决什么

第 30 节解决了“测什么”：只前向或前向、损失、反向。本节解决“测多大的模型和输入”：把配置作为命令行选项，避免每次修改源文件，也让日志记录实验实际条件。

### 每一步做了什么，为什么这样做

1. 将词表大小、上下文长度、模型宽度、层数、头数、前馈维度、RoPE theta、批大小和实际序列长度开放为参数。模型构造、随机输入和目标、输出形状检查必须一起使用这些参数，避免只更改入口却留下硬编码。
2. 增加 MODEL_CONFIGS 和 --model-size。当前预设有 small、medium、large、xl、10B，默认 small。预设存在不等于这些模型都已在本机验证。
3. 为结构覆盖参数使用 None 默认值。未指定时继承预设，明确指定时覆盖对应字段，便于区分“没填”与“填了一个数”。
4. 复制选中的预设后再覆盖，避免修改原始预设字典。覆盖后的结果写回 args，日志与模型构造使用同一份最终数值。
5. 结构字段实际偏离预设时显示 custom (based on ...)，同时打印 overrides 和实际结构。相同数值的显式覆盖不会标记为 custom。该标记只针对四个结构字段；词表、输入等条件仍须看完整日志，不能只看 small 标签。
6. 合并配置之后、创建模型之前验证最终参数。这样可以及早给出可理解的错误，而不是等到模型内部计算失败。
7. 打印实际配置、模型参数 dtype、设备、张量形状和梯度追踪状态。目标的 int64 代表 token 编号类型，不能用来判断模型浮点精度。
8. 应用户明确要求清理重复打印：保留创建模型前的配置汇总，以及后面的设备、dtype、形状和训练状态检查；删除重复的模式、模型配置、输入配置打印。计时和计算逻辑未修改。

### 参数检查及原因

| 检查 | 为什么需要 |
| --- | --- |
| 尺寸参数均大于零 | 避免空尺寸、无效层数或除零 |
| d_model 能被 num_heads 整除 | 模型宽度要能均分到各注意力头 |
| 每头维度为偶数 | 当前 RoPE 实现按成对维度旋转 |
| sequence_length 不超过 context_length | 当前 RoPE 缓存仅按上下文长度建立 |
| rope_theta 是有限正数 | 避免非法旋转频率参数，包括 NaN、无穷大、零与负数 |

初次检查发现 rope_theta 只排除了小于等于零，不能排除 NaN 和正无穷；学生已修改为有限正数检查。复查同时确认模型 dtype 日志已补充。上述结论是源码检查；用户尚未提供非法参数输入、结构覆盖等分支的运行结果，不能把它们记成已实测通过。

### 本次实测配置：默认已不再是玩具尺寸

| 项目 | 本次值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| 模型预设 | small |
| d_model / d_ff | 768 / 3072 |
| num_layers / num_heads | 12 / 12 |
| vocab_size | 10000 |
| context_length / sequence_length | 512 / 512 |
| batch_size | 4 |
| 参数 dtype | torch.float32 |
| 模型与输入设备 | cuda:0 |
| 输入与目标形状 | [4, 512] |
| 输出形状 | [4, 512, 10000] |
| 模型训练状态 / 梯度追踪 | True / True |
| 预热 / 正式测量 | 5 / 10 次 |

用户分别运行了模块 cs336_systems.new_benchmark，选项为 --mode forward --model-size small，以及 --mode forward_backward --model-size small。

两种模式均完成首轮检查、全部预热和正式测量，无报错。前向模式目标编号范围 10–9997；前后向模式为 2–9987，均位于合法范围。损失的具体数值本次没有打印。反向检查针对选定输出层参数，不代表全面验证所有梯度的数值正确性。

### 原始耗时（毫秒）

| 轮次 | 前向预热 | 前向正式测量 | 前向＋损失＋反向预热 | 前向＋损失＋反向正式测量 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 240.264100 | 89.328100 | 503.923300 | 275.315200 |
| 2 | 98.533900 | 87.724400 | 272.872600 | 274.118600 |
| 3 | 94.443900 | 88.473500 | 275.590800 | 275.468800 |
| 4 | 88.732300 | 88.873600 | 273.969400 | 273.810500 |
| 5 | 89.851100 | 90.575400 | 276.042500 | 275.469700 |
| 6 | — | 88.558300 | — | 273.840000 |
| 7 | — | 90.311300 | — | 275.792600 |
| 8 | — | 89.083800 | — | 274.993800 |
| 9 | — | 88.066300 | — | 275.796400 |
| 10 | — | 89.478100 | — | 278.530600 |

| 指标 | 只前向 | 前向＋损失＋反向 |
| --- | ---: | ---: |
| 平均耗时（ms） | 89.047280 | 275.313620 |
| 总体标准差（ms，除以 N） | 0.866575 | 1.296965 |
| 相对标准差 | 约 0.97% | 约 0.47% |
| 最小耗时（ms） | 87.724400 | 273.810500 |
| 最大耗时（ms） | 90.575400 | 278.530600 |

### 如何解释，而不误读结果

- small 两种模式均已实际运行通过，参数和日志切换生效。本组样本的相对波动较小，可作为当前环境的初步性能基线，不等于跨运行的稳定性已经充分验证。
- 合计均值约为前向的 3.09 倍，是本次实验观察，不是固定理论比例。
- 均值差包含损失计算、反向及两次独立运行的差异，不能直接称为精确的反向耗时。
- 两次启动未固定随机种子，不能声称权重和随机输入逐元素相同。
- 前向保持梯度追踪并建立计算图，不是关闭梯度追踪的推理耗时。
- 清理上一轮梯度、初始化、打印和检查均在计时外。前后向模式仍未执行优化器更新，不是完整训练一步。
- 首轮预热显著偏慢，但没有混入统计。仅凭日志不能确定具体开销来源。
- 第 30 节的玩具配置约为 5 ms / 12 ms；本次模型与输入都增大，不能将耗时增加解释为代码性能退化。
- 本次成功只证明该配置在用户这次运行中可执行，不能外推其他预设都能装入显存或已完成全部作业实验。

### 下一步学习实验（尚未执行）

保持 small 模型、batch_size=4、context_length=512 和精度不变，仅将实际 sequence_length 分别设为 128、256，两种模式各测一次，并与本次 512 对比。这样每次只改变一个主要工作量因素，观察实际序列长度与耗时的关系。最大上下文长度与实际输入长度是不同概念，前者可以保持 512。

### 当前学生代码快照

以下是记录时从本地 new_benchmark.py 原样读取的代码，包括已清理的日志和末尾未执行的旧验证字符串；未新增作业实现。


```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse

# model configuration
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", 
    choices = ["forward", "forward_backward"],
    default = "forward",
    help = "forward: forward pass only, forward_backward: forward pass + loss + backward pass"
    )
# model configuration
parser.add_argument(
    "--model-size",
    choices=list(MODEL_CONFIGS),
    default="small",
)

parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--rope-theta", type=float, default=10000.0)

# if None is specified, the configuration will be inherited from the model size
parser.add_argument("--d-model", type=int, default=None)
parser.add_argument("--num-layers", type=int, default=None)
parser.add_argument("--num-heads", type=int, default=None)
parser.add_argument("--d-ff", type=int, default=None)

# input configuration
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--sequence-length", type=int, default=512)

args = parser.parse_args()

# check the configuration
config = MODEL_CONFIGS[args.model_size].copy()
overrides = {}

for name in config:
    value = getattr(args, name)

    if value is not None:
        if value != config[name]:
            overrides[name] = value
        config[name] = value

    # write the final value back to args
    setattr(args, name, config[name])

if overrides:
    config_name = f"custom (based on {args.model_size})"
else:
    config_name = args.model_size


mode = args.mode
vocab_size = args.vocab_size
context_length = args.context_length
batch_size = args.batch_size
sequence_length = args.sequence_length

if mode == "forward":
    measurement_name = "forward time"
elif mode == "forward_backward":
    measurement_name = "forward pass + loss + backward pass time"
else:
    raise ValueError(f"Invalid mode: {mode}")

# 4. check the final configuration
sizes = [
    vocab_size,
    context_length,
    batch_size,
    sequence_length,
    args.d_model,
    args.d_ff,
    args.num_layers,
    args.num_heads,
]

if any(size <= 0 for size in sizes):
    parser.error("model and input sizes must be greater than 0")

if args.d_model % args.num_heads != 0:
    parser.error("--d-model must be divisible by --num-heads")

if (args.d_model // args.num_heads) % 2 != 0:
    parser.error("when using the current RoPE implementation, the dimension of each attention head must be even")

if sequence_length > context_length:
    parser.error("--sequence-length must be less than or equal to --context-length")

if not (0 < args.rope_theta < float("inf")):
    parser.error("--rope-theta must be a finite positive number")



# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))
print("measurement mode: ", mode)
print("model size: ", config_name)
print("actual model structure: ", config)

if overrides:
    print("overrides: ", overrides)

print("vocab size: ", vocab_size)
print("maximum context length: ", context_length)
print("RoPE theta: ", args.rope_theta)
print("batch size: ", batch_size)
print("actual sequence length: ", sequence_length)

# 2. create model
model = BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)
model = model.to(device)
model.train()

# keep gradient tracking for both modes
assert torch.is_grad_enabled()

# 3. create random input and target

x = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < vocab_size)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("model parameters dtype:", next(model.parameters()).dtype)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs
def run_steps():
    logits = model(x)
    loss = None
    if mode == "forward_backward":
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="mean",
        )
        loss.backward()
    return logits, loss

warmup_runs = 5
measurement_runs = 10

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)


    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits, loss = run_steps()
    #=======================


    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(
    f"warmup run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, vocab_size)
        assert logits.device == x.device 

        if mode == "forward_backward":
            assert loss is not None
            assert loss.shape == torch.Size([])
            assert loss.requires_grad
            assert torch.isfinite(loss).item()
            assert loss.device == targets.device

            param = model.lm_head.weight
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert torch.isfinite(param.grad).all().item()
            del param 

        print("the present mode check passed")
    # release the output of the warmup run
    del logits, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()

    logits, loss = run_steps()

    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(
    f"formal measurement run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )
    # release the output of the measurement run
    del logits, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"measurement name: {measurement_name}")
print(f"average time of {measurement_name}: {average_time:.6f} milliseconds")
print(f"standard deviation of {measurement_name}: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of {measurement_name}: {min(time_ms):.6f} milliseconds")
print(f"max time of {measurement_name}: {max(time_ms):.6f} milliseconds")


""" # 9. two forward and backward verification, no timing
print("\n two forward and backward verification, no timing starts...")

# select the output layer weights for observation
param = model.lm_head.weight
assert param.requires_grad

for i in range(2):
    print(f"\n this is the {i + 1} forward and backward verification")

    # ① clear the gradient
    model.zero_grad(set_to_none=True)

    print("the gradient after clearing: ", param.grad)
    assert param.grad is None

    # ② re-execute the forward pass for each round
    logits = model(x)

    assert logits.shape[:2] == targets.shape
    assert logits.device == targets.device

    # ③ reshape the logits and targets, and re-calculate the loss for each round
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert loss.device == x.device
    assert loss.requires_grad

    print("the loss of this round: ", loss.item())

    # ④ perform the backward propagation for the loss of this round
    loss.backward()

    # ⑤ check the gradient
    assert param.grad is not None
    assert param.grad.shape == param.shape
    assert torch.isfinite(param.grad).all().item()

    print("the gradient after backward propagation exists: ", param.grad is not None)
    print("the shape of the parameter: ", param.shape)
    print("the shape of the gradient: ", param.grad.shape)
    print("the gradient finite value check passed")

    # ⑥ after the check of this round, release the output and loss of this round
    del logits, flat_logits, flat_targets, loss

# after the two forward and backward verifications, clear the gradient of the last round
model.zero_grad(set_to_none=True)
del param

print("\n two forward and backward verification, no timing ends...") """

```

### 用户提供的原始运行日志

```text
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> python -m cs336_systems.new_benchmark --mode forward --model-size small
if cuda is available: True
GPU device:  NVIDIA GeForce RTX 3060 Ti                                                   
measurement mode:  forward                                                                
model size:  small                                                                        
actual model structure:  {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12}
vocab size:  10000                                                                        
maximum context length:  512                                                              
RoPE theta:  10000.0                                                                      
batch size:  4                                                                            
actual sequence length:  512                                                              
target shape: torch.Size([4, 512])
target device: cuda:0     
target dtype: torch.int64 
target maximum index: 9997
target minimum index: 10  
model parameters device: cuda:0      
model parameters dtype: torch.float32
input device: cuda:0
input shape: torch.Size([4, 512])
model is training: True
gradient tracking enabled: True

 warmup starts...
warmup run 1: forward time: 240.264100 milliseconds
output device: cuda:0
output shape: torch.Size([4, 512, 10000])
the present mode check passed
warmup run 2: forward time: 98.533900 milliseconds
warmup run 3: forward time: 94.443900 milliseconds
warmup run 4: forward time: 88.732300 milliseconds
warmup run 5: forward time: 89.851100 milliseconds

 formal measurement starts...
formal measurement run 1: forward time: 89.328100 milliseconds
formal measurement run 2: forward time: 87.724400 milliseconds
formal measurement run 3: forward time: 88.473500 milliseconds
formal measurement run 4: forward time: 88.873600 milliseconds
formal measurement run 5: forward time: 90.575400 milliseconds
formal measurement run 6: forward time: 88.558300 milliseconds
formal measurement run 7: forward time: 90.311300 milliseconds
formal measurement run 8: forward time: 89.083800 milliseconds
formal measurement run 9: forward time: 88.066300 milliseconds
formal measurement run 10: forward time: 89.478100 milliseconds

 statistics starts...
measurement name: forward time
average time of forward time: 89.047280 milliseconds
standard deviation of forward time: 0.866575 milliseconds
measurement runs: 10
min time of forward time: 87.724400 milliseconds
max time of forward time: 90.575400 milliseconds
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> python -m cs336_systems.new_benchmark --mode forward_backward --model-size small
if cuda is available: True
GPU device:  NVIDIA GeForce RTX 3060 Ti
measurement mode:  forward_backward
model size:  small
actual model structure:  {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12}
vocab size:  10000
maximum context length:  512
RoPE theta:  10000.0
batch size:  4
actual sequence length:  512
target shape: torch.Size([4, 512])
target device: cuda:0
target dtype: torch.int64
target maximum index: 9987
target minimum index: 2
model parameters device: cuda:0
model parameters dtype: torch.float32
input device: cuda:0
input shape: torch.Size([4, 512])
model is training: True
gradient tracking enabled: True

 warmup starts...
warmup run 1: forward pass + loss + backward pass time: 503.923300 milliseconds
output device: cuda:0
output shape: torch.Size([4, 512, 10000])
the present mode check passed
warmup run 2: forward pass + loss + backward pass time: 272.872600 milliseconds
warmup run 3: forward pass + loss + backward pass time: 275.590800 milliseconds
warmup run 4: forward pass + loss + backward pass time: 273.969400 milliseconds
warmup run 5: forward pass + loss + backward pass time: 276.042500 milliseconds

 formal measurement starts...
formal measurement run 1: forward pass + loss + backward pass time: 275.315200 milliseconds
formal measurement run 2: forward pass + loss + backward pass time: 274.118600 milliseconds
formal measurement run 3: forward pass + loss + backward pass time: 275.468800 milliseconds
formal measurement run 4: forward pass + loss + backward pass time: 273.810500 milliseconds
formal measurement run 5: forward pass + loss + backward pass time: 275.469700 milliseconds
formal measurement run 6: forward pass + loss + backward pass time: 273.840000 milliseconds
formal measurement run 7: forward pass + loss + backward pass time: 275.792600 milliseconds
formal measurement run 8: forward pass + loss + backward pass time: 274.993800 milliseconds
formal measurement run 9: forward pass + loss + backward pass time: 275.796400 milliseconds
formal measurement run 10: forward pass + loss + backward pass time: 278.530600 milliseconds

 statistics starts...
measurement name: forward pass + loss + backward pass time
average time of forward pass + loss + backward pass time: 275.313620 milliseconds
standard deviation of forward pass + loss + backward pass time: 1.296965 milliseconds
measurement runs: 10
min time of forward pass + loss + backward pass time: 273.810500 milliseconds
max time of forward pass + loss + backward pass time: 278.530600 milliseconds
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> 

```


## 32. 加入 AdamW 更新：train 模式实现、验证与实测

记录日期：2026-09-20。来源为本次对话、当前学生源文件及用户提供的 Windows / RTX 3060 Ti 原始日志。助手只做源码检查与数据整理，没有运行 GPU 测试，没有改写训练实现。

### 后续笔记记录约定

用户明确要求：每次代码检查、修改或提供新运行结果后，均同步更新本笔记，记录“做了什么、为什么这样做、检查结果和下一步”，并保存当次代码快照。静态检查、用户实测、未执行的建议要分别标明。保留旧记录，便于复习演进过程；不能把当前快照自动当作 Windows 当时运行文件的逐字证明。

### 为什么加入 optimizer.step()

此前 backward 只计算并累加参数梯度，不会自行改变权重。优化器使用梯度更新权重后，才形成包含参数更新的训练计算路径。因此保留原有两种模式，并增加 train；序列长度实验不是加入优化器之前的必要步骤。

| 模式 | 每轮计时内的计算 |
| --- | --- |
| forward | 前向，保留梯度追踪 |
| forward_backward | 前向、平均交叉熵损失、反向 |
| train | 前向、平均交叉熵损失、反向、AdamW 更新 |

train 标签明确注明 excluding zero_grad。这里的“完整训练一步”指上述计算路径，仍不包含数据加载、清梯度、日志等完整应用开销。

### 每一步做了什么，以及为什么

1. 在模式选项、帮助信息和测量标签中增加 train。保留旧模式才能继续分别观察各类工作负载。
2. 在模型移到 GPU 后、循环外创建一次 PyTorch AdamW。优化器持有模型参数，并保存历史状态；若每轮重新创建，会丢失状态，不能代表正常连续训练。
3. 仅 train 创建优化器；当前学习率为 0.001、weight_decay 为 0.0，其余选项采用库默认值。日志记录实际优化器及这些显式设置。该性能结果对应 PyTorch 实现，不能直接当作其他 AdamW 实现的结果。
4. run_steps 中先前向、算损失、反向，再在 train 模式更新参数。本轮梯度必须在更新前保留，不能在 backward 与 step 之间清掉。
5. 每轮开始时清梯度并同步，再开始计时。这样上一轮梯度不累加进本轮，同时延续“清理梯度不计入”的约定。当前优化器使用全部模型参数，因此通过模型清梯度覆盖了这些参数。
6. 预热与正式测量都调用同一函数，train 预热也更新参数和优化器状态。首次状态初始化进入预热；每轮结束前同步，确保参数更新在停止计时前完成。
7. 首轮 train 预热开始计时前，对输出层权重执行 detach 后 clone，保留独立旧值。detach 本身不复制存储，clone 才防止旧值随原参数一起变化；不应只用另一个变量名引用同一参数。
8. 停止计时后检查新权重有限、是否至少有一个元素改变，并记录最大绝对变化。复制和比较位于计时外，不把验证操作算进正式工作量。
9. 保留损失、梯度的形状、存在性与有限性检查，当前快照还检查输出 requires_grad。所选输出层的基本检查不等于全部参数或优化器数学正确性已全面验证。
10. 增加 --warmup-steps，默认 5，允许零并拒绝负数。零预热用于研究首次运行开销，此时不应偷偷先执行一次完整训练步“补验证”，否则实验条件已经改变。
11. 根据复查建议，当前文件已增加零预热提示：预热阶段正确性检查不会执行，train 的首次正式测量包含优化器状态初始化。输入、配置等循环前检查仍会执行。零预热分支尚无用户运行结果。
12. 当前文件已把“本模式检查通过”移到参数更新检查之后，避免提前宣布通过。原始运行日志也显示参数变化结果在通过提示之前。
13. 每轮释放输出和损失引用，结束后清理最后一轮梯度；均位于计时外。正式结果只统计 10 个测量样本，标准差使用总体口径（除以 N）。

### 先静态检查，再确认实测

首次源码审查确认：优化器在循环外、更新在反向之后、两种循环覆盖相同路径、复制旧值的方法正确、同步和计时边界正确。发现的日志问题是零预热未提示跳过验证、通过提示早于更新验证，当前快照均已处理。

用户随后运行 cs336_systems.new_benchmark，使用 --mode train --model-size small。该次运行成功完成，不是助手本地 GPU 测试。日志开头夹带的 forward_backward 片段位于 train 启动命令之前，不能混入本次 train 的样本。

### 本次实验配置与更新验证

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| 模型 | small：宽度 768，前馈维度 3072，12 层，12 头 |
| 词表 / 最大上下文 | 10000 / 512 |
| 批大小 / 实际序列长度 | 4 / 512 |
| 输入与目标 / 输出 | [4,512] / [4,512,10000] |
| 参数 dtype / 设备 | float32 / cuda:0 |
| 模型训练状态 / 梯度追踪 | True / True |
| 优化器 / 学习率 / 权重衰减 | PyTorch AdamW / 0.001 / 0.0 |
| 目标编号范围 | 1–9991，int64 |
| 预热 / 正式测量 | 5 / 10 次 |
| 参数是否改变 | True |
| 所选权重最大绝对变化 | 9.99998301e-04，约 0.001 |

最大变化量接近学习率是本次观察，不能据此认为所有参数都改变了相同数值，也不能单凭这一数值证明 AdamW 全部计算正确。本次损失数值未打印。随机输入和随机目标用于性能测量，不以每步损失都下降作为更新成功的必要条件。

### 原始耗时（ms）

| 轮次 | train 预热 | train 正式测量 |
| --- | ---: | ---: |
| 1 | 546.721700 | 296.833800 |
| 2 | 294.999500 | 297.703000 |
| 3 | 295.412900 | 297.597100 |
| 4 | 296.373900 | 298.880600 |
| 5 | 297.148800 | 298.120800 |
| 6 | — | 299.454800 |
| 7 | — | 302.226500 |
| 8 | — | 302.388900 |
| 9 | — | 298.350400 |
| 10 | — | 298.554200 |

均值 299.011010 ms，总体标准差 1.784693 ms，最小 296.833800 ms，最大 302.388900 ms，相对标准差约 0.60%。5 个预热样本没有纳入统计。

### 与此前 small 结果对照

| 模式 | 平均耗时（ms） | 总体标准差（ms） | 来源 |
| --- | ---: | ---: | --- |
| 只前向 | 89.047280 | 0.866575 | 第 31 节 |
| 前向、损失、反向 | 275.313620 | 1.296965 | 第 31 节 |
| 前向、损失、反向、参数更新 | 299.011010 | 1.784693 | 本节 |

三种模式均有成功运行记录，可作为当前环境的初步基线。train 与之前前后向均值相差 23.697390 ms（约 23.70 ms），这不是单独隔离测得的 optimizer.step 耗时：还包含跨运行的状态、初始化及噪声差异，不能直接当作精确优化器耗时。

### 必须记住的实验含义

- 预热不是“假训练”：5 次 train 预热已经改变权重和优化器状态，正式测量第一轮是第 6 次更新，最后一轮是第 15 次更新。
- forward_backward 不更新权重；train 每轮更新权重。两者不是在每一轮都保持完全相同数值状态的实验。
- 输入和目标在循环外生成并重复使用，目的是控制工作量，不代表真实数据集训练流程。
- 首轮预热慢，但仅凭日志不能把差额精确归因于某种初始化开销。
- 本组相对波动较小，不等于多次独立运行的稳定性已充分验证。
- 本次 small 配置成功不代表所有更大模型均已实测或都能装入显存。
- 尚未实测零预热分支；不要把源码中的提示当作该分支的运行证明。

### 下一步建议（尚未执行）

可保持 small 模型、batch_size=4、context_length=512 和精度不变，对实际 sequence_length=128、256、512 做单变量对比，并明确每份结果对应哪一种模式。后续也可以研究零预热对统计的影响，但一次实验先改变一个主要条件。

### 当次代码完整快照

以下是记录时读取的学生文件原文，包含零预热提示、输出梯度追踪检查以及调整后的检查通过提示。此次笔记更新没有修改该代码。

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse

# model configuration
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", 
    choices = ["forward", "forward_backward", "train"],
    default = "forward",
    help = (
        "forward: forward pass only;" 
        "forward_backward: forward pass + loss + backward pass;"
        "train: forward pass + loss + backward pass + parameter update"
        ),
    )
# model configuration
parser.add_argument(
    "--model-size",
    choices=list(MODEL_CONFIGS),
    default="small",
)

parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--rope-theta", type=float, default=10000.0)

# if None is specified, the configuration will be inherited from the model size
parser.add_argument("--d-model", type=int, default=None)
parser.add_argument("--num-layers", type=int, default=None)
parser.add_argument("--num-heads", type=int, default=None)
parser.add_argument("--d-ff", type=int, default=None)

# input configuration
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--sequence-length", type=int, default=512)

parser.add_argument(
    "--warmup-steps",
    type=int,
    default=5,
    help="number of warmup steps; may be 0",
)

args = parser.parse_args()

if args.warmup_steps < 0:
    parser.error("--warmup-steps must be greater than or equal to 0")

# check the configuration
config = MODEL_CONFIGS[args.model_size].copy()
overrides = {}

for name in config:
    value = getattr(args, name)

    if value is not None:
        if value != config[name]:
            overrides[name] = value
        config[name] = value

    # write the final value back to args
    setattr(args, name, config[name])

if overrides:
    config_name = f"custom (based on {args.model_size})"
else:
    config_name = args.model_size


mode = args.mode
vocab_size = args.vocab_size
context_length = args.context_length
batch_size = args.batch_size
sequence_length = args.sequence_length

if mode == "forward":
    measurement_name = "forward time"
elif mode == "forward_backward":
    measurement_name = "forward pass + loss + backward pass time"
elif mode == "train":
    measurement_name = "forward pass + loss + backward pass + parameter update time (excluding zero_grad)"
else:
    raise ValueError(f"Invalid mode: {mode}")

# 4. check the final configuration
sizes = [
    vocab_size,
    context_length,
    batch_size,
    sequence_length,
    args.d_model,
    args.d_ff,
    args.num_layers,
    args.num_heads,
]

if any(size <= 0 for size in sizes):
    parser.error("model and input sizes must be greater than 0")

if args.d_model % args.num_heads != 0:
    parser.error("--d-model must be divisible by --num-heads")

if (args.d_model // args.num_heads) % 2 != 0:
    parser.error("when using the current RoPE implementation, the dimension of each attention head must be even")

if sequence_length > context_length:
    parser.error("--sequence-length must be less than or equal to --context-length")

if not (0 < args.rope_theta < float("inf")):
    parser.error("--rope-theta must be a finite positive number")



# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))
print("measurement mode: ", mode)
print("model size: ", config_name)
print("actual model structure: ", config)

if overrides:
    print("overrides: ", overrides)

print("vocab size: ", vocab_size)
print("maximum context length: ", context_length)
print("RoPE theta: ", args.rope_theta)
print("batch size: ", batch_size)
print("actual sequence length: ", sequence_length)

# 2. create model
model = BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)
model = model.to(device)
model.train()

# optimizer
optimizer = None

if mode == "train":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )
    print("optimizer:", type(optimizer).__name__)
    print("learning rate:", optimizer.param_groups[0]["lr"])
    print("weight decay:", optimizer.param_groups[0]["weight_decay"])

# keep gradient tracking enabled for all three modes
assert torch.is_grad_enabled()

# 3. create random input and target

x = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < vocab_size)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("model parameters dtype:", next(model.parameters()).dtype)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs
def run_steps():
    logits = model(x)
    loss = None
    if mode in ("forward_backward", "train"):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        loss.backward()

        if mode == "train":
            optimizer.step()
    return logits, loss


warmup_runs = args.warmup_steps
measurement_runs = 10

print("warmup steps:", warmup_runs)

if warmup_runs == 0:
    print("note: no correctness check for warmup steps")

    if mode == "train":
        print("note: the first measurement round will include the initialization of the optimizer state")

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)
    # only save the old parameters in the first warmup run for train mode
    if mode == "train" and i == 0:
        param_before = model.lm_head.weight.detach().clone()
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)


    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits, loss = run_steps()
    #=======================


    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(
    f"warmup run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, vocab_size)
        assert logits.device == x.device 
        assert logits.requires_grad


        if mode in ("forward_backward", "train"):
            assert loss is not None
            assert loss.shape == torch.Size([])
            assert loss.requires_grad
            assert torch.isfinite(loss).item()
            assert loss.device == targets.device

            param = model.lm_head.weight
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert torch.isfinite(param.grad).all().item()
            del param 

        # check if the parameters have changed in train mode
        if mode == "train":
            param_after = model.lm_head.weight.detach()

            assert torch.isfinite(param_after).all().item()

            changed = (param_after != param_before).any().item()
            max_change = (param_after - param_before).abs().max().item()

            assert changed, "the selected parameters have not changed"

            print("parameter changed:", changed)
            print(f"maximum absolute change: {max_change:.8e}")

            del param_before, param_after
        print("the present mode check passed")
    # release the output of the warmup run
    del logits, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()

    logits, loss = run_steps()

    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(
    f"formal measurement run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )
    # release the output of the measurement run
    del logits, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"measurement name: {measurement_name}")
print(f"average time of {measurement_name}: {average_time:.6f} milliseconds")
print(f"standard deviation of {measurement_name}: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of {measurement_name}: {min(time_ms):.6f} milliseconds")
print(f"max time of {measurement_name}: {max(time_ms):.6f} milliseconds")




```

### 用户原始日志（包含 train 启动前的旧输出片段）

```text
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> python -m cs336_systems.new_benchmark --mode forward_backward --model-size small
if cuda is available: True
GPU device:  NVIDIA GeForce RTX 3060 Ti
measurement mode:  forward_backward
model size:  small
actual model structure:  {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12}
vocab size:  10000
maximum context length:  512
RoPE theta:  10000.0
batch size:  4
actual sequence length:  512
target shape: torch.Size([4, 512])
target device: cuda:0
target dtype: torch.int64
target maximum index: 9987
target minimum index: 2
model parameters device: cuda:0
model parameters dtype: torch.float32
input device: cuda:0
input shape: torch.Size([4, 512])
model is training: True
gradient tracking enabled: True

 warmup starts...
warmup run 1: forward pass + loss + backward pass time: 503.923300 milliseconds
output device: cuda:0
output shape: torch.Size([4, 512, 10000])
the present mode check passed
warmup run 2: forward pass + loss + backward pass time: 272.872600 milliseconds
warmup run 3: forward pass + loss + backward pass time: 275.590800 milliseconds
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> python -m cs336_systems.new_benchmark --mode train --model-size small
if cuda is available: True
GPU device:  NVIDIA GeForce RTX 3060 Ti                                                   
measurement mode:  train                                                                  
model size:  small                                                                        
actual model structure:  {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12}
vocab size:  10000                                                                        
maximum context length:  512                                                              
RoPE theta:  10000.0                                                                      
batch size:  4                                                                            
actual sequence length:  512                                                              
optimizer: AdamW
learning rate: 0.001              
weight decay: 0.0                 
target shape: torch.Size([4, 512])
target device: cuda:0             
target dtype: torch.int64         
target maximum index: 9991        
target minimum index: 1           
model parameters device: cuda:0      
model parameters dtype: torch.float32
input device: cuda:0                 
input shape: torch.Size([4, 512])    
model is training: True              
gradient tracking enabled: True
warmup steps: 5

 warmup starts...
warmup run 1: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 546.721700 milliseconds
output device: cuda:0
output shape: torch.Size([4, 512, 10000])
parameter changed: True
maximum absolute change: 9.99998301e-04
the present mode check passed
warmup run 2: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 294.999500 milliseconds
warmup run 3: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 295.412900 milliseconds
warmup run 4: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 296.373900 milliseconds
warmup run 5: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 297.148800 milliseconds

 formal measurement starts...
formal measurement run 1: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 296.833800 milliseconds
formal measurement run 2: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 297.703000 milliseconds
formal measurement run 3: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 297.597100 milliseconds
formal measurement run 4: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 298.880600 milliseconds
formal measurement run 5: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 298.120800 milliseconds
formal measurement run 6: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 299.454800 milliseconds
formal measurement run 7: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 302.226500 milliseconds
formal measurement run 8: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 302.388900 milliseconds
formal measurement run 9: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 298.350400 milliseconds
formal measurement run 10: forward pass + loss + backward pass + parameter update time (excluding zero_grad): 298.554200 milliseconds

 statistics starts...
measurement name: forward pass + loss + backward pass + parameter update time (excluding zero_grad)
average time of forward pass + loss + backward pass + parameter update time (excluding zero_grad): 299.011010 milliseconds
standard deviation of forward pass + loss + backward pass + parameter update time (excluding zero_grad): 1.784693 milliseconds
measurement runs: 10
min time of forward pass + loss + backward pass + parameter update time (excluding zero_grad): 296.833800 milliseconds
max time of forward pass + loss + backward pass + parameter update time (excluding zero_grad): 302.388900 milliseconds
(cs336-systems) PS C:\Users\Administrator\cs336\Assignment02> 

```


## 33. 下一步计划：序列长度对比实验（尚未执行）

三种模式在 small、batch_size=4、sequence_length=512 下已有基线。下一步先不修改代码，利用现有参数控制单一变量，理解输入工作量与耗时的关系。

### 固定什么，改变什么，为什么

保持 small 模型结构、词表 10000、context_length=512、batch_size=4、float32、5 次预热和 10 次测量不变，只改变实际 sequence_length。最大上下文长度是当前 RoPE 缓存支持的上限，实际序列长度才是本轮输入长度；测 128 或 256 时不需要同步缩小最大上下文长度。

第一步选择 forward 模式、sequence_length=128。预期输入和目标形状 [4,128]，输出 [4,128,10000]，日志显示 context_length=512、sequence_length=128。先确认这些维度和配置，再解释计时。该计划只使用已有参数，本次无源码修改，代码快照沿用第 32 节。

之后测 forward 的 256；再补齐 forward_backward 和 train 的 128、256，与既有 512 结果对照。保持 GPU 工作环境尽量一致；现有 512 来自先前运行，若驱动、代码、精度或运行环境改变，需要重测基线。各次独立启动未固定随机种子，不能声称数值输入或权重完全相同。

### 待填写记录（均值 ± 总体标准差，ms）

| 实际序列长度 | forward | forward_backward | train（排除 zero_grad） |
| --- | --- | --- | --- |
| 128 | 待测 | 待测 | 待测 |
| 256 | 待测 | 待测 | 待测 |
| 512 | 89.047280 ± 0.866575 | 275.313620 ± 1.296965 | 299.011010 ± 1.784693 |

### 学习问题

观察序列长度翻倍时，各模式耗时增长多少，不预设必须是两倍或四倍。模型中不同部分对序列长度的计算量依赖不同，且耗时还受调度、内存访问等影响；整体测量不能单独证明某一部分的占比。先收集结果，再做解释。所有新长度结果目前均未执行，不填写预测数字充当测量值。


## 34. 对照用户贴出的 (a) 交付要求：当前完成度

本次直接依据用户提供的 (a) 原文和当前源码核对，不扩大为整个第 2 章要求。未修改脚本或执行新 GPU 测试。

| (a) 要求 | 当前状态 |
| --- | --- |
| 给定超参数初始化 basics Transformer | 已实现，支持模型预设和结构参数覆盖 |
| 生成随机批数据 | 已实现，随机整数输入与目标随配置变化 |
| 预热 w 步后再计时 | 已实现，--warmup-steps 默认 5，预热不进入正式样本 |
| 计时 n 步 | 已实现 n=10，但 measurement_runs 仍固定在源码中 |
| 参数选择只前向、前后向、含优化器更新 | 已实现 forward / forward_backward / train |
| 使用高精度计时 | 使用 timeit.default_timer |
| 每步后 CUDA 同步 | 预热和正式循环均在工作完成后同步，再读取结束时间 |
| 交付脚本 | 当前 new_benchmark.py 已具备核心功能；三种模式有用户提供的 small 运行记录 |

结论：就 (a) 而言，核心功能已经基本完成，不需要先完成序列长度实验才能满足这一项。第 33 节的长度实验是额外性能探索，不是该段交付要求的前置条件。

最值得补齐的是把正式测量次数 n 也开放为参数，并验证为正整数，避免 n=0 导致均值、最小值等统计无法定义；默认仍可保留 10。原文明确要求模式由参数选择，但没有明确规定 w、n 都必须是命令行参数，因此不能把固定 n=10 武断判为不合格。参数化 n 是为了更完整、便利地支持通用 w/n 测量。

(a) 的具体模式列表要求组合路径，不要求另写“只反向”或“只 optimizer.step”的单独模式。用户提供的这一段也未指定必须使用自写优化器；当前 PyTorch AdamW 满足该段参数更新的功能要求，其他章节若有专门规定需另行核对。

train 的计时明确排除 zero_grad，包含前向、损失、反向、优化器更新；该段文字没有明确要求把清梯度纳入计时，保持当前已声明口径即可。无需因“完整训练”标签而擅自改变计时定义。

下一步优先补 n 的可配置性，并用小的正整数确认实际正式轮数和统计次数一致、零和负数会被拒绝；此项目前未实现或实测。没有新增源码，代码快照沿用第 32 节。


## 35. 正式测量次数 n 参数化：源码检查通过

记录日期：2026-09-21。本次读取学生最新代码并静态检查，没有修改源码或执行 GPU 测试。

### 每一步做了什么，为什么

1. 第 34–39 行定义 --measurement-steps，类型为整数，默认 10。将正式采样次数开放到命令行，同时保持原来默认行为。
2. 第 66–67 行拒绝小于等于零的值，位置在配置处理、CUDA 检查和模型创建之前。避免没有样本时除零或无法求最小、最大值；非整数由 argparse 的整数解析拒绝。
3. 第 240–243 行分别读取预热和测量次数，并打印两者。w 与 n 独立，不会因为调整正式轮数而改变预热。
4. 第 325 行正式循环使用 measurement_runs，每轮追加一个时间样本。不存在残留的固定 10 次循环。
5. 第 357–365 行均值、总体标准差、样本数使用 time_ms 的实际长度，最小和最大值也取自同一列表。统计与用户选择的次数一致。
6. 三种模式仍共用上述循环，预热与正式测量都在每步后同步，计时边界未被本次改动改变。

### 结论及验证边界

正式测量次数参数已正确接入，静态检查没有发现此次改动的问题。就用户提供的 (a) 条款而言，脚本现在具备超参数初始化、随机输入、可选三种路径、可配置 w/n、计时及每步同步等要求的功能。此前三种模式已有 small 配置运行记录；本次新增的自定义 n 分支尚无运行日志，不能标为实测通过。

建议最后在用户 GPU 环境做一次 --measurement-steps 3 的短运行，确认恰好 3 条正式测量和最终 measurement runs: 3；默认预热仍为 5。再检查 0 和负数被参数检查拒绝。不指定时按源码默认仍为 10。无需为完成 (a) 继续增加模式或性能实验；性能实验可在后续条目开展。

### 当次学生代码完整快照

以下为本次检查时的文件原文，未改写实现。

```python
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse

# model configuration
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", 
    choices = ["forward", "forward_backward", "train"],
    default = "forward",
    help = (
        "forward: forward pass only;" 
        "forward_backward: forward pass + loss + backward pass;"
        "train: forward pass + loss + backward pass + parameter update"
        ),
    )
# model configuration
parser.add_argument(
    "--model-size",
    choices=list(MODEL_CONFIGS),
    default="small",
)
# measurement steps
parser.add_argument(
    "--measurement-steps",
    type=int,
    default=10,
    help="number of timed steps",
)

parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--rope-theta", type=float, default=10000.0)

# if None is specified, the configuration will be inherited from the model size
parser.add_argument("--d-model", type=int, default=None)
parser.add_argument("--num-layers", type=int, default=None)
parser.add_argument("--num-heads", type=int, default=None)
parser.add_argument("--d-ff", type=int, default=None)

# input configuration
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--sequence-length", type=int, default=512)

parser.add_argument(
    "--warmup-steps",
    type=int,
    default=5,
    help="number of warmup steps; may be 0",
)

args = parser.parse_args()

if args.warmup_steps < 0:
    parser.error("--warmup-steps must be greater than or equal to 0")
if args.measurement_steps <= 0:
    parser.error("--measurement-steps must be greater than 0")

# check the configuration
config = MODEL_CONFIGS[args.model_size].copy()
overrides = {}

for name in config:
    value = getattr(args, name)

    if value is not None:
        if value != config[name]:
            overrides[name] = value
        config[name] = value

    # write the final value back to args
    setattr(args, name, config[name])

if overrides:
    config_name = f"custom (based on {args.model_size})"
else:
    config_name = args.model_size


mode = args.mode
vocab_size = args.vocab_size
context_length = args.context_length
batch_size = args.batch_size
sequence_length = args.sequence_length

if mode == "forward":
    measurement_name = "forward time"
elif mode == "forward_backward":
    measurement_name = "forward pass + loss + backward pass time"
elif mode == "train":
    measurement_name = "forward pass + loss + backward pass + parameter update time (excluding zero_grad)"
else:
    raise ValueError(f"Invalid mode: {mode}")

# 4. check the final configuration
sizes = [
    vocab_size,
    context_length,
    batch_size,
    sequence_length,
    args.d_model,
    args.d_ff,
    args.num_layers,
    args.num_heads,
]

if any(size <= 0 for size in sizes):
    parser.error("model and input sizes must be greater than 0")

if args.d_model % args.num_heads != 0:
    parser.error("--d-model must be divisible by --num-heads")

if (args.d_model // args.num_heads) % 2 != 0:
    parser.error("when using the current RoPE implementation, the dimension of each attention head must be even")

if sequence_length > context_length:
    parser.error("--sequence-length must be less than or equal to --context-length")

if not (0 < args.rope_theta < float("inf")):
    parser.error("--rope-theta must be a finite positive number")



# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))
print("measurement mode: ", mode)
print("model size: ", config_name)
print("actual model structure: ", config)

if overrides:
    print("overrides: ", overrides)

print("vocab size: ", vocab_size)
print("maximum context length: ", context_length)
print("RoPE theta: ", args.rope_theta)
print("batch size: ", batch_size)
print("actual sequence length: ", sequence_length)

# 2. create model
model = BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)
model = model.to(device)
model.train()

# optimizer
optimizer = None

if mode == "train":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )
    print("optimizer:", type(optimizer).__name__)
    print("learning rate:", optimizer.param_groups[0]["lr"])
    print("weight decay:", optimizer.param_groups[0]["weight_decay"])

# keep gradient tracking enabled for all three modes
assert torch.is_grad_enabled()

# 3. create random input and target

x = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < vocab_size)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("model parameters dtype:", next(model.parameters()).dtype)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs
def run_steps():
    logits = model(x)
    loss = None
    if mode in ("forward_backward", "train"):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        loss.backward()

        if mode == "train":
            optimizer.step()
    return logits, loss


warmup_runs = args.warmup_steps
measurement_runs = args.measurement_steps
print("measurement steps:", measurement_runs)
print("warmup steps:", warmup_runs)

if warmup_runs == 0:
    print("note: no correctness check for warmup steps")

    if mode == "train":
        print("note: the first measurement round will include the initialization of the optimizer state")

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)
    # only save the old parameters in the first warmup run for train mode
    if mode == "train" and i == 0:
        param_before = model.lm_head.weight.detach().clone()
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)


    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits, loss = run_steps()
    #=======================


    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(
    f"warmup run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, vocab_size)
        assert logits.device == x.device 
        assert logits.requires_grad


        if mode in ("forward_backward", "train"):
            assert loss is not None
            assert loss.shape == torch.Size([])
            assert loss.requires_grad
            assert torch.isfinite(loss).item()
            assert loss.device == targets.device

            param = model.lm_head.weight
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert torch.isfinite(param.grad).all().item()
            del param 

        # check if the parameters have changed in train mode
        if mode == "train":
            param_after = model.lm_head.weight.detach()

            assert torch.isfinite(param_after).all().item()

            changed = (param_after != param_before).any().item()
            max_change = (param_after - param_before).abs().max().item()

            assert changed, "the selected parameters have not changed"

            print("parameter changed:", changed)
            print(f"maximum absolute change: {max_change:.8e}")

            del param_before, param_after
        print("the present mode check passed")
    # release the output of the warmup run
    del logits, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()

    logits, loss = run_steps()

    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(
    f"formal measurement run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )
    # release the output of the measurement run
    del logits, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"measurement name: {measurement_name}")
print(f"average time of {measurement_name}: {average_time:.6f} milliseconds")
print(f"standard deviation of {measurement_name}: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of {measurement_name}: {min(time_ms):.6f} milliseconds")
print(f"max time of {measurement_name}: {max(time_ms):.6f} milliseconds")




```


## 36. 远程 Windows 提示不认识 --measurement-steps

记录日期：2026-09-21。用户已进入 C:\Users\Administrator\cs336\Assignment02 并激活 .venv，执行 train / small / --measurement-steps 3 后，argparse 报 unrecognized arguments。帮助列表包含 --warmup-steps，但没有 --measurement-steps。

### 判断及原因

报错发生在参数解析阶段，尚未开始模型初始化或 GPU 计时，不是 CUDA 或显存问题。命令与第 35 节检查的本地接口一致；远程实际执行版本未注册该参数。最可能是最新源文件尚未同步到远程，也可能是 Python 导入了另一份同名模块；仅凭报错不能确定是哪一种。

### 排查步骤与理由

1. 在远程 PowerShell 用 Get-Location 确认项目目录。用 Select-String 在 .\cs336_systems\new_benchmark.py 中搜索 measurement-steps 与 args.measurement_steps，核对该文件是否包含参数定义和使用。
2. 如果找不到，用用户已有的文件同步方式把本地最新 new_benchmark.py 更新到远程项目对应位置。先留意远程独有改动，避免覆盖；不需要重装 PyTorch 或虚拟环境。
3. 运行模块的 --help。看到 --measurement-steps 后，说明实际执行的参数解析器支持它，再运行 3 步测试。
4. 若远程文件已包含该参数但 --help 仍没有，检查 where.exe python 与 $env:PYTHONPATH，继续排查解释器或同名模块来源，不盲目重装环境。

本次是根据错误日志作出的诊断，尚未确认文件同步完成，也未取得 n=3 实测结果。本地源码未改动，代码快照仍见第 35 节。后续应记录远程搜索结果、帮助输出和运行结果。
