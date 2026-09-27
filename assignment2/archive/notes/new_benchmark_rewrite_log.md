# 重写 benchmark：仅前向（进行中）

- 日期：2026-09-19
- 脚本：现已移到 `archive/new_benchmark.py`（重写用，当时不直接改旧的 `benchmark.py`，现为 `benchmark_lm.py`）
- 机器：Mac 写代码；台式机 `PC-20230218RALO`（RTX 3060 Ti）跑 GPU
- Git：`150d934` 只提交了这份新脚本
- 对应作业：第 2 章 2.1.1–2.1.3，先做 **forward-only**

这份笔记记的是**已经做过的步骤和原因**，方便以后复习。后面的计时循环还没写，见文末「还没做」。

---

## 0. 为什么要重写，而不是继续改旧脚本

旧的 `cs336_systems/benchmark.py` 已经能跑三种模式、NVTX、checkpoint、`torch.compile`。重写的目的不是再堆功能，而是：

1. 把「模型能跑」和「时间测得对」拆开，一步只加一个概念。
2. 自己再走一遍：创模型 → 看设备 → 上 GPU → 预热 → 同步计时。
3. 旧脚本太大，出错时分不清是模型问题、设备问题，还是计时口径问题。

作业真正要交的是一套可复用脚本，三种模式都要有。实现顺序仍应是：**先让仅前向的计时可信，再挂 loss / backward / optimizer**。

---

## 1. 先做正确性冒烟，不先计时

### 做了什么

用 `BasicsTransformerLM` 建一个很小的模型（4 层，`d_model=512`，`context_length=128`），随机 token 输入 `batch=2`、`seq=16`，跑一次 `model(x)`，检查 `logits` 形状是 `(B, T, vocab_size)`。

### 为什么

- 作业测的是效率，可以用随机权重、随机输入，不必先训出好模型。
- 形状错了再谈毫秒没有意义。
- 先用小配置：台式机显存有限，逻辑错了也能立刻看出来，不会一上来 OOM。

这一步只证明：**导入路径对、模型能前向、输出形状对**。当时默认设备是 CPU。

---

## 2. 决定从「仅前向」开始测

一次训练步可以拆成：Forward → Loss → Backward → Optimizer。作业三种模式就是这四段的前缀。

先只测前向，是因为：

- 计时环（预热、同步、均值/标准差）和「测哪一段」是两件事，不要同时调试。
- 前向是后面所有模式的公共部分。
- 训练前向应开着梯度（`model.train()`），以便以后和 backward 比；`torch.no_grad()` 是另一条「推理前向」，数字会偏快，不能直接拿来解释「加了反向变慢多少」。

**还没做的计时口径**（写循环时再落实）：

- 默认预热 5 步 + 正式测量 10 步；预热和测量走同一条前向，预热时间丢掉。
- 初始化、随机输入、清梯度不进计时。
- 每步：`synchronize` → 开始计时 → 前向 → `synchronize` → 结束计时。
- GPU 异步：不同步的话，停表时往往只测到 CPU 提交 kernel，不是 GPU 算完。

预热的原因：第一次调用常有 CUDA context、显存分配、cuBLAS 选算法等一次性成本。不预热时第一步会明显偏慢，均值和标准差都被拉歪。作业后面还要对比 warmup = 0 / 1 / 2 / 5。

---

## 3. 为什么要连台式机，不在 Mac 上测

Mac 这份环境没有可用的作业级 CUDA GPU。作业要求测的是 **GPU 算完的时间**。

台式机上次 small 实验用过：

- 路径：`C:\Users\Administrator\cs336\Assignment02`
- 环境：`.venv\Scripts\python.exe`（`torch 2.11.0+cu128`）
- **不要**在该目录 `uv run` / `uv sync`：会按仓库 `uv.lock` 把 CUDA 版卸掉、装回 CPU 版。

### 连接方式

`~/.ssh/config` 里有别名 `desktop-pc`。DHCP 会换 IP，本次从旧的 `192.168.8.4` 变成了 `192.168.8.5`。Mac 终端：

```bash
ssh desktop-pc
```

看到 `PS C:\Users\Administrator>` 才是 Windows。OpenSSH 较旧时会有 post-quantum 警告，可忽略。

进入旧环境：

```powershell
cd C:\Users\Administrator\cs336\Assignment02
.\.venv\Scripts\Activate.ps1
```

提示符出现 `(.venv)` 后再用 `python`，不要用 `uv run`。

`Activate.ps1` 若被策略拦住，先：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 改代码和跑代码为什么要分开

Cursor 当前窗口编的是 **Mac 上的仓库**。台式机有一份独立的 `Assignment02`，不会自动更新。流程是：

1. Mac 上改 `new_benchmark.py`
2. 拷到台式机同名路径
3. 在已激活的 `.venv` 里跑 `python -m cs336_systems.new_benchmark`

从 **Mac 本机另开终端** 拷（不要在已经 ssh 进去的 PowerShell 里 scp）：

```bash
scp /Users/xiaofei/01-LLM/project/05CS336/Assignment02/cs336_systems/new_benchmark.py \
  desktop-pc:"C:/Users/Administrator/cs336/Assignment02/cs336_systems/new_benchmark.py"
```

本次实际由助手执行同步。拷完用哈希或 `Get-Content` 确认是新文件。

---

## 4. 上 GPU 之前，先打印设备（不搬模型）

### 做了什么

在创建 `model`、`x` 之后、`model(x)` 之前增加三项打印：

1. `torch.cuda.is_available()`：这套 PyTorch **能不能**用 CUDA。
2. `next(model.parameters()).device`：第一个参数张量在哪。单卡、未做模型并行时，可代表整网。
3. `x.device`：输入在哪。和看 `x.shape` 一样，只是属性不同。

### 为什么要插在前向之前

设备和输入不一致时，前向会先报错，诊断信息就看不到了。  
`is_available() == True` **不等于**模型和输入已经在 GPU 上。`.to("cuda")` 不会自动发生。

对照：

| CUDA 可用 | 模型参数 | 输入 | 含义 |
|-----------|----------|------|------|
| True | cpu | cpu | 能用 GPU，这轮仍在 CPU 上算 |
| True | cuda:0 | cuda:0 | 都在第一张 GPU 上，可以前向 |
| True | cuda:0 | cpu（或反过来） | 设备不匹配，不能直接 `model(x)` |
| False | cpu | cpu | 只能先做 CPU 检查，先查驱动 / 是否进错环境 |

未搬设备时，台式机上预期是第一行：`True` / `cpu` / `cpu`。这是成功的诊断，不是失败。

`model.parameters()` 是迭代器，不能直接 `.device`，所以用 `next(...)` 取第一个参数。

中间曾把前向和 shape 打印写了两遍，后来删掉一组，避免输出重复。

---

## 5. 确认诊断之后，再把模型和输入搬到同一块 GPU

### 做了什么

```text
model = model.to("cuda:0")
x = x.to("cuda:0")
```

设备打印放在这两行**之后**，再执行前向和 shape 断言。

### 为什么

- 作业要测的是 GPU 时间，必须先在 GPU 上算出正确结果。
- 模型和输入必须在**同一设备**，否则 einsum / 线性层会对不上。
- 仍用小模型：这一步只验证「搬得动、前向过、形状对」，不换 small，也不加计时。
- `cuda:0` 表示当前可见的第一张 GPU。台式机单卡时就是那张 3060 Ti。

搬完后，三行应接近：`True`（若还保留可用性打印）、`cuda:0`、`cuda:0`，然后 shape 断言通过。

当前脚本里可用性打印后来拿掉了，只留参数设备和输入设备。逻辑上已经默认「准备在 GPU 上跑」；若某次 `to("cuda:0")` 失败，错误会出在搬设备，而不是前向。

---

## 6. 提交 git 时只收这一份新脚本

提交 `150d934`：`Add a small forward smoke script to verify CUDA and tensor devices.`

只加了 `cs336_systems/new_benchmark.py`。当时工作区里还有 flash attention、writeup、nsys、results 等，和这次「从零重写仅前向」无关，所以没放进同一次提交。

没有 `git push`。

---

## 7. 当前脚本停在哪

已经具备：

- 小模型 + 随机输入
- 搬到 `cuda:0`
- 打印设备和输出形状
- 一次前向 + 断言

还没有：

- 预热 5 + 测量 10
- `torch.cuda.synchronize()` 包住的计时
- 均值和总体标准差（如 `statistics.pstdev`）
- 作业表 1 的 small（768 / 3072 / 12 层 / 12 头，batch 4，context 512）
- loss / backward / AdamW
- 命令行切换模式
- NVTX / Nsight / 混合精度

---

## 8. 下次从哪里继续

建议仍一次只加一层，在 Mac 改、再同步到台式机跑：

1. 保持小模型，加上预热 / 测量循环和同步计时，打印 10 个原始毫秒数、均值、标准差。
2. 用两个对照判断计时有没有写对：拿掉同步（时间会异常短）；warmup=0（第一个样本会明显偏大）。
3. 数字稳定后再换成 small。
4. 同一套计时环上再挂 loss、backward、optimizer。

计时里约定：训练前向、开着梯度；不要把 `zero_grad`、数据生成算进去。

判断「仅前向计时写对了」的经验：有预热时 10 个数接近，标准差远小于均值；第一步不应再特别慢。

---

## 9. 复习时容易混的几句话

1. **能用 CUDA** 和 **正在 GPU 上算** 是两件事。
2. **CPU 调用返回** 和 **GPU 算完** 是两件事。后者才是作业要的时间。
3. **第一次偏慢** 多半是预热问题，不一定是模型稳态就那么慢。
4. **训练前向** 和 **推理前向** 数字不能混着比。
5. 台式机跑实验用 `.venv` 的 `python`，不要 `uv run`。
6. Mac 上的文件和台式机上的文件不是同一份，改完必须同步再跑。
