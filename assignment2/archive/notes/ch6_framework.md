# 第 6 章框架：优化器状态分片

一句话：DDP 之后，每张卡仍各存一份完整 AdamW 状态。这一章让每张卡只更新、只保存大约 `1 / world_size` 的参数动量，更新完再把参数广播回去。参数和梯度仍然完整复制。

两道题，共 20 分。

| 题 | 分 | 要交什么 | 在哪做 |
| --- | ---: | --- | --- |
| `optimizer_state_sharding` | 15 | 分片优化器类，测试稳定通过 | `cs336_systems/optimizer_state_sharding.py`，适配器 `tests/adapters.py` 的 `get_sharded_optimizer` |
| `optimizer_state_sharding_accounting` | 5 | (a) 峰值显存和拆账 (b) 每步时间 (c) 和 ZeRO stage 1 的差别 | `cs336_systems/benchmark_optimizer_sharding.py`，写进 writeup |

先做正确性，再测显存和速度。测试不需要 GPU。

## 1. 它接在第 5 章的哪里

一次训练步在两张卡上可以拆成：

| 阶段 | 第 5 章 DDP 已经做的 | 第 6 章新增的 |
| --- | --- | --- |
| 前向 | 每卡一份完整参数，各算自己的数据 | 不变 |
| 反向 | 每卡得到自己的梯度 | 不变 |
| 梯度同步 | `all-reduce` 再除以 `world_size`，梯度变完整且一致 | 这一章不改梯度 |
| 优化器 | 每卡用完整梯度更新全部参数，并保存全部动量 | 每卡只更新自己负责的参数，只保存那一部分动量 |
| 参数对齐 | 因为大家做了同一次更新，参数自然一致 | 更新范围不同，必须把更新后的参数再同步一次 |

省掉的是 AdamW 状态，不是参数，也不是梯度。第 7 章 FSDP 才继续切参数和梯度。

## 2. 显存里到底有什么

AdamW 对每个参数存两个 FP32 状态：一阶动量 `exp_avg`，二阶动量 `exp_avg_sq`。状态体积是参数的 2 倍。

xl（词表 10,000，`d_model=2560`，32 层，32 头，`d_ff=10240`，embedding 与 `lm_head` 不共享）参数量：

`3,406,809,600`

FP32 下一份参数是 **12.691 GiB**。两张卡、`world_size=2` 时，静态部分（不含激活）是：

| 组成部分 | 不分片 | 分片后 | 是否被这一章切开 |
| --- | ---: | ---: | --- |
| 参数 | 12.691 GiB | 12.691 GiB | 否 |
| 梯度 | 12.691 GiB | 12.691 GiB | 否 |
| AdamW 状态 `m+v` | 25.383 GiB | 12.691 GiB | 是，大约减半 |
| 合计 | 50.766 GiB | 38.074 GiB | 少一份状态，12.691 GiB |

激活、反向保存的张量、CUDA 缓存不在这张表里。所以测出来的峰值会高于 38 GiB / 51 GiB，两边的差值应主要来自优化器状态。

PyTorch 的 AdamW 在第一次 `step()` 才创建动量。因此：

- 初始化之后：两边都主要是参数，应接近。
- `step()` 之前、反向刚结束：两边都加上完整梯度，外加这一步的激活峰值，优化器状态还没有。
- 第一次 `step()` 之后：不分片多出约 25.4 GiB 状态，分片多出约 12.7 GiB。

`benchmark_optimizer_sharding.py` 目前只在开头 `reset_peak_memory_stats()` 一次，后面打印的是从进程开始起的历史峰值。前向激活的高水位会盖住后面的优化器分配，`before_step` 和 `after_step` 可能印出同一个数。对照上面三个时刻时，每个时刻要同时看当前占用量，并在进入下一阶段前重置峰值。

## 3. 类要完成的三件事

讲义要求包一层任意 `torch.optim.Optimizer`，自己继承 `torch.optim.Optimizer`。`__init__` 里必须调用父类构造函数。

```text
ShardedOptimizer
  外层 param_groups：全部参数，zero_grad() 能清掉每张卡上的完整梯度
  内层 optimizer：只含本 rank 的参数，动量只在这里创建
  param -> owner rank：step 之后知道向谁要新参数
```

`add_param_group` 在父类构造时就会被调用，训练中途加组时也会被调用。分配必须写在这里，不能只写在 `__init__`。

分配规则用轮询，按参数张量而不是按元素：

```text
owner = 全局参数序号 % world_size
```

本 rank 等于 owner 的参数放进内层优化器。某一组在本 rank 上一个参数都没有时，不要建空组。

`step` 的顺序固定：

1. 若本 rank 有参数，调用内层 `optimizer.step(closure, **kwargs)`。
2. 对每个参数 `broadcast(param.data, src=owner)`。
3. 返回内层 `step` 的 loss。没有本地参数时，loss 用 `None`，广播仍然要做。

测试在 CPU、Gloo、2 进程上跑。`ToyModel` 和 `ToyModelWithTiedWeights` 各 10 步，分片优化器和普通 AdamW 的最终权重必须一致。建议连跑 5 次。

```bash
uv run pytest tests/test_sharded_optimizer.py
```

测试不包 DDP。随机种子相同，两张卡的输入和梯度本来就一样，所以它只检查「局部更新 + 广播」是否等价于「每人更新全部参数」。`fc4.weight = fc2.weight` 这种绑定，在 `parameters()` 里只出现一次，按张量对象分配即可。`ToyModel` 里还有 `requires_grad=False` 的参数，它仍在参数列表中，只是没有梯度、不会被改写。

## 4. 计分题 (a)(b)(c) 各自在问什么

标准配置与第 5 章相同：1 机 2 卡，xl，全局 batch 的口径要在 writeup 里写明。

**(a) 显存。** 三个时刻：初始化后、`optimizer.step` 前、`optimizer.step` 后。两边各测一次。用第 2 节的表解释差值落在 `m+v`，参数和梯度不变。激活峰值两边应接近，因为它发生在优化器状态创建之前。

**(b) 速度。** 同一配置，预热之后量每步墙钟。分片后每张卡少做约一半参数更新，但多了一次参数广播。writeup 要写出两边的毫秒数，并说明时间变了是因为少了更新还是多了通信。

**(c) 和 ZeRO stage 1 的差别。** 讲义点名 ZeRO-DP \(P_{os}\)（Rajbhandari et al., 2020）。对照时抓住两处：

| | 本作业 | ZeRO stage 1 |
| --- | --- | --- |
| 优化器状态 | 按参数张量切开 | 同样切开 |
| 梯度 | 每卡仍保留完整梯度（DDP 的 `all-reduce` 之后） | 梯度按同一分片 `reduce-scatter`，每卡只留下自己那一份 |
| 参数同步 | 每个参数从 owner `broadcast` | 一次 `all-gather` 把更新后的参数拼齐 |
| 通信量 | 在 DDP 的梯度 `all-reduce` 之外，再多传一轮更新后的参数 | 用「梯度 `reduce-scatter` + 参数 `all-gather`」替换原来的梯度 `all-reduce`，每步通信量与普通 DDP 同量级 |
| 显存 | 只少优化器状态 | 优化器状态和梯度一起少 |

所以这一章是 stage 1 的简化版：省状态，不省梯度，通信是额外加上去的。

## 5. 和现有脚本的关系

`optimizer_state_sharding.py` 里已有外层优化器、轮询分片、内层 AdamW 和 `step` 后 `broadcast` 的骨架。先用测试确认它和普通 AdamW 等价，再改测量脚本。

`benchmark_optimizer_sharding.py` 现在两张卡各跑完整模型，没有套第 5 章的 DDP。它能比较「本地 AdamW」和「分片优化器 + 广播」，但不是「DDP 训练步」对「DDP + 分片优化器」。写 (a)(b) 之前先定这个口径，并在答案里写出来。

## 6. 做题顺序

1. 读懂第 2 节的三笔账：参数、梯度、`m+v`。
2. 跑通 `tests/test_sharded_optimizer.py`，连跑 5 次。
3. 改测量脚本，使三个时刻的当前占用和峰值能分开。
4. 在 2×GPU 上跑 xl 的显存和计时。
5. 用第 4 节的表写 (c)，两到三句，只比较显存和通信量。
