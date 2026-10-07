# Safety Supplement 任务分解表

来源：`cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`（v26.0.0）
用途：把 PDF 中散落在各 Problem 里的 deliverable 汇总成可勾选的任务清单，每项给出**验收判据**（自检标准）与**常见翻车点**。

> 本表只描述「做什么」和「怎么判断做对了」，不含实现方案。代码由你写。

---

## 0. 全局约定（在写任何脚本前先定死）

| 项 | Supplement 要求 | 说明 |
|---|---|---|
| 基座模型 | `meta-llama/Meta-Llama-3.1-8B`（base，非 instruct） | §3/§5 的 zero-shot 与 SFT 评测都用它 |
| 裁判模型 | `meta-llama/Llama-3.3-70B-Instruct` | AlpacaEval annotator + `evaluate_safety.py` |
| 解码（§3、§5 全部） | greedy：`temperature=0.0`, `top_p=1.0` | 与主 RL 作业的 `temperature=1.0` 不同，必须分开 |
| §3 提示词 | `prompts_safety/zero_shot_system_prompt.prompt` | 四类任务统一用同一个 system prompt |
| §5 提示词 | `prompts_safety/alpaca_sft.prompt` | **§5 不要再叠 zero-shot system prompt** |
| 停止条件 | 生成到 `# Query:` 即停止 | 只对 §3 的 zero-shot system prompt 生效 |
| 禁用 | Hugging Face `Trainer` 类 | 训练脚本必须自己写循环 |
| 共享卷 | 模型/数据只读 `/mnt/cs336-a5-supplement`；结果写 `/mnt/cs336-a5-supplement-results` | 不要写共享卷 |

**建议的目录约定**：新建 `results_safety/`，与主 RL 作业的 `results/` 完全隔离（两套实验的模型、提示词、解码参数都不同，混在一起后续对比必错）。

---

## 1. 进度总表

| 编号 | PDF Problem | 交付物类型 | 分值 | 依赖 | 状态 |
|---|---|---|---|---|---|
| T0.1 | 全局 | 生成路径基建 | — | — | ☐ |
| T1.1–T1.6 | `mmlu_baseline` | 代码 + 实验 + 文字 | 4 | T0.1 | ☐ |
| T2.1–T2.6 | `gsm8k_baseline` | 代码 + 实验 + 文字 | 4 | T0.1 | ☐ |
| T3.1–T3.4 | `alpaca_eval_baseline` | 代码 + 实验 + 文字 | 4 | T0.1 | ☐ |
| T4.1–T4.4 | `sst_baseline` | 代码 + 实验 + 文字 | 4 | T0.1 | ☐ |
| T5.1 | `look_at_sft` | 纯文字 | 4 | 数据可达 | ☐ |
| T6.1–T6.2 | `data_loading` | 代码 | 3 | — | ✅ 模板已过 |
| T7.1–T7.3 | `sft_script` + `sft` | 代码 + 训练 + 文字 | 4 + 6 | T6 | ☐ |
| T8.1–T8.5 | `mmlu_sft` / `gsm8k_sft` / `alpaca_eval_sft` / `sst_sft` | 实验 + 文字 | 4×4 | T7 | ☐ |
| T9.1 | `red_teaming` | 纯文字 | 4 | T7 | ☐ |
| T10.1–T10.2 | `look_at_hh` | 代码 + 文字 | 2 | — | ☐ |
| T11.1 | `dpo_loss` | 代码 | 2 | — | ✅ 模板已过 |
| T12.1–T12.4 | `dpo_training` + 三处评测 | 代码 + 训练 + 文字 | 4 | T7, T10, T11 | ☐ |

「✅ 模板已过」= 上游 `tests/adapters.py` 中已实现且实测通过，但 **deliverable 属于你自己的模块仍需存在**（见 T6.3、T11.2）。

---

## 2. 逐任务明细

### T0.1 生成路径基建（无独立分值，但阻塞 §3 与 §5 全部）

**做什么**：一条与 GRPO 完全隔离的「greedy 生成 + 落盘」路径。

**验收判据**
- 存在一个可复用的函数：输入 prompt 列表 → 输出 (texts, elapsed_seconds)，且 `elapsed_seconds` 只计生成阶段。
- 采样参数可显式传入，默认满足 `temperature=0.0, top_p=1.0`。
- §3 与 §5 的差异只体现在「prompt 怎么拼」，生成与落盘逻辑共用。
- 落盘产物包含：原始 example、拼好的 prompt、模型 generation、解析结果、score、耗时。

**常见翻车点**
- 直接改 `prompting_data_construction.py` 的默认 `temperature` → 会让主 RL 作业的历史结果不可复现。**新增参数或新增模块，不要改旧默认值。**
- 把 vLLM 首次加载/编译时间算进吞吐 → 吞吐数字偏小。计时应该只包住 `generate` 调用。

---

### §3.1 MMLU baseline（4 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T1.1 | 解析函数 | `tests/adapters.py` 中的实现已通过 2 个测试；额外自测 `**The correct answer is B**`、`The correct answer is: B`、小写形式，记录各自返回什么 |
| T1.2 | 评测脚本 | 能在 10 条样例上跑通；产物含 example/generation/parse/score |
| T1.3 | 全量运行 + 吞吐 | 全量 MMLU；报出 `N/seconds`；给出与解析失败条数 |
| T1.4 | 结果分析 (e) | 1–2 句给出准确率数字 |
| T1.5 | 错例分析 (f) | 随机 10 条错例 + 2–4 句错误类型归纳 |
| T1.6 | 失败解析分析 (c) | 报出失败条数；非零则贴几条原文 |

**验收命令**：`.venv/bin/python -m pytest tests/test_metrics.py -q`

**常见翻车点**
- `mmlu_zero_shot.prompt` 里是 `{options[0]}`…`{options[3]}`，`str.format` 处理不了 `[0]` → 拼 prompt 时会抛 `KeyError`/`IndexError`。**先把一条样例的最终字符串打印出来确认。**
- 嵌套顺序反了（应该：task prompt → 作为 `{instruction}` → system prompt）。
- 解析器要求字面 `The correct answer is X`；模型输出 Markdown 加粗时词边界仍成立，但加冒号就失败 → 这正是 (c) 的答案来源。

---

### §3.2 GSM8K baseline（4 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T2.1 | 解析函数 | 已通过 2 个测试；额外确认多位数被完整取出（如输出以 `72.` 结尾不会退化成 `2`） |
| T2.2 | 评测脚本 | prompt 用 `prompts_safety/gsm8k_zero_shot.prompt`，**不经过** `prompts/question_only.prompt` |
| T2.3 | 全量运行 + 吞吐 | 报出 `N/seconds` |
| T2.4 | 结果分析 (e) | 1–2 句准确率 |
| T2.5 | 错例分析 (f) | 10 条错例 + 2–4 句 |
| T2.6 | 失败解析分析 (c) | 条数 + 例子 |

**常见翻车点（PDF 专门加了 Note，是最高频的错）**
- 复用主 RL 作业里「取 `####` 之后」的解析逻辑 → PDF §3.2 明确说这里用不同的 parser（取最后一个数字）和不同的提示词。
- 提示词里没有 `####`、没有 `\boxed{}`，模型可能长篇推理 → `max_tokens` 太小会截断，导致「最后一个数字」根本不是最终答案。
- 千分位 `1,000` 之类的数字，想想会解析成什么。

---

### §3.3 AlpacaEval baseline（4 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T3.1 | 预测脚本 | 输出 **JSON array**（非 JSONL），每项含 `instruction` / `output` / `generator` / `dataset` |
| T3.2 | 吞吐 (b) | `N/seconds` |
| T3.3 | winrate (c) | `alpaca_eval` 跑通；报 **winrate + length-controlled winrate** 两个数 |
| T3.4 | 错例分析 (d) | 从 annotations 里取 `preference == 1.0` 的 10 条 + 2–4 句，含「是否同意自动评估」 |

**验收命令**（PDF 原文）
```
alpaca_eval --model_outputs <path_to_model_predictions.json> \
    --reference_outputs data/alpaca_eval/alpaca_eval_gpt4_turbo.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \
    --base-dir .
```

**常见翻车点**
- 序列化成 JSONL → alpaca_eval 直接读不了。
- `generator` 字段在数组内不一致 → 会被当成多个模型。
- `dataset` 字段硬编码 → 先确认 `alpaca_eval_gpt4_turbo.json` 里这个键的真实名称。
- 跑完没检查 `scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json` 是否生成（(d) 要从这里取数据）。

---

### §3.4 SimpleSafetyTests baseline（4 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T4.1 | 预测脚本 | 输出 **JSONL**，每行至少含 `prompts_final` 与 `output` |
| T4.2 | 吞吐 (b) | `N/seconds` |
| T4.3 | 安全比例 (c) | 用 `scripts/evaluate_safety.py` 得到 safe 比例 |
| T4.4 | 错例分析 (d) | 10 条 judged unsafe + 2–4 句 |

**验收命令**（PDF 原文）
```
uv run python scripts/evaluate_safety.py \
    --input-path <path_to_model_predictions.jsonl> \
    --model-name-or-path /mnt/cs336-a5-supplement/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path <path_to_write_output.jsonl>
```

**常见翻车点**
- `scripts/evaluate_safety.py:52` 硬读 `example["prompts_final"]` → 键名不匹配会 KeyError。**先看 CSV 的真实列名再决定映射。**
- `--model-name-or-path` 写成 HF id（应指向共享卷路径）。
- 输出 jsonl 里 `metrics["safe"] == 0.0` 表示被判不安全 → (d) 的筛选条件。

---

### §4.1 `look_at_sft`（4 分，纯文字）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T5.1 | 抽 10 条训练样本并撰写分析 | 2–4 句，指出出现的传统 NLP 任务类型 + 对 prompt/response 质量的具体评论（带原文例子） |

**数据路径**：`cs336_alignment/modal_utils_safety.py` 中的 `SFT_TRAIN_PATH`（`train.jsonl.gz`）。

---

### §4.2.1 `data_loading`（3 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T6.1 | Packed Dataset | `pytest -k test_packed_sft_dataset` 通过 |
| T6.2 | Batching | `pytest -k test_iterate_batches` 通过 |
| T6.3 | 把实现搬进 `cs336_alignment/`（可选但推荐） | deliverable 是「你的」Dataset，不是测试文件里的内部类 |

**验收命令**：`.venv/bin/python -m pytest tests/test_data.py -q` → 已实测 2 passed

**需要能口头解释的点**（答辩/自查用）
- `len(stream) - 1` 里的 `-1` 为什么存在？（labels 相对 input 右移一位）
- 文档间分隔符是什么、为什么必须有？
- 末尾不足 `seq_length` 的 token 怎么处理、丢了多少信息？
- `shuffle=True` 的随机源是什么、能否复现？

---

### §4.2.2 `sft_script` + `sft`（4 + 6 分，约 3 B200 hrs）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T7.1 | 数据加载接入训练 | 训练 loss 在合理范围（不是 NaN、不是常数） |
| T7.2 | 训练脚本 | 支持配置超参、梯度累积、周期性 train/val 日志；未使用 `Trainer` |
| T7.3 | 完成训练 | 1 epoch、ctx 512、有效 batch 32、lr 2e-5、cosine、warmup 3%、wd 0.1、clip 1.0（可自行调 lr 做实验） |
| T7.4 | 保存产物 | `save_pretrained` 同时保存 model + tokenizer，供 §5 和 §6 复用 |

**硬要求核对**
- 加载：`torch_dtype=torch.bfloat16`, `attn_implementation="flash_attention_2"`
- loss：`F.cross_entropy`，注意 logits/labels 的形状变换与是否需要 `ignore_index`
- 梯度累积：**loss 先除以 k 再 backward**，每 k 步 `step()` + `zero_grad()`

**自查实验**
- 把 k 调成 1 与 k 调成 4，验证「同样有效 batch 下 loss 曲线一致」→ 证实除以 k 的位置正确。
- 验证集 loss 的算法必须与训练一致（同一 packing、同一形状变换），否则曲线不可比。

**交付物**：训练设置描述 + final validation loss + 学习曲线 + 序列化模型/tokenizer。现有 `scripts/plot_training_metrics.py` 的绘图习惯可沿用。

---

### §5 SFT 模型评测（4×4 = 16 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T8.1 | §5.1 MMLU | 吞吐 + 准确率 + 与 zero-shot 对比 + 10 条错例定性分析 |
| T8.2 | §5.2 GSM8K | 同上 |
| T8.3 | §5.3 AlpacaEval | 吞吐 + winrate + LC winrate + 与 baseline 对比 + 10 条 dispreferred 分析 |
| T8.4 | §5.4 SimpleSafetyTests | 吞吐 + safe 比例 + 与 baseline 对比 + 10 条 unsafe 分析 |
| T8.5 | 红队（§5.5，独立 4 分） | 见 T9.1 |

**输入格式规则（PDF §5 开头明确）**
- **不要**再用 `zero_shot_system_prompt.prompt`。
- MMLU / GSM8K：先格式化 `mmlu_zero_shot.prompt` / `gsm8k_zero_shot.prompt`，把结果作为 instruction 放进 `alpaca_sft.prompt`。
- AlpacaEval / SST：instruction 直接放进 `alpaca_sft.prompt`。
- 解析器、序列化格式、`evaluate_safety.py` 命令**全部不变**，只换 prompt 构造。

**常见翻车点**
- 忘记「§5 不叠 system prompt」→ 与训练格式不一致，性能虚低。
- 用 SFT 前的脚本跑 SFT 模型却忘了换 checkpoint 路径。

---

### §5.5 `red_teaming`（4 分，纯文字）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T9.1a | 三个**新**的滥用场景 | 1–3 句，不能重复 PDF 已列的「协助危险活动」 |
| T9.1b | 三个恶意应用的实际尝试 | 每个 2–4 句：方法、是否成功、尝试时长、所用策略 |

---

### §6.2 `look_at_hh`（2 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T10.1 | HH 加载函数 | 合并 4 个文件；丢弃多轮；拆出 instruction + chosen/rejected；记录来源文件 |
| T10.2 | 数据观察 | 3 条 helpful + 3 条 harmless 的差异归纳 + 是否同意标注 |

**文件**：`data/hh/` 下 `harmless-base`、`helpful-base`、`helpful-online`、`helpful-rejection-sampled`（均为 `.jsonl.gz`）。

**自查实验**
- 打印「丢弃的多轮样本数 / 保留数」比例，确认过滤逻辑不是把全部样本都丢了（或一条都没丢）。
- 抽样打印 chosen/rejected 全文自己读，而不是只看长度统计。

---

### §6.3 `dpo_loss`（2 分）

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T11.1 | 单实例 DPO loss | `pytest -k test_per_instance_dpo_loss` 通过（期望值 0.9104） |
| T11.2 | 搬到 `cs336_alignment/`（可选） | deliverable 是「你写的」函数 |

**验收命令**：`.venv/bin/python -m pytest tests/test_dpo.py -q` → 已实测 1 passed

**需要能解释的点**
- 为什么可以只算 response 段的 log-prob 而忽略 prompt？（PDF §6.3 给了理由）
- EOS 拼接方式对 tokenization 的影响（测试里那个「response 必须是 prompt 的续接」断言就是为了抓这个）。
- loss 应返回在哪个 device 上、`lm_ref` 在另一张卡时怎么处理。
- β=0 与 β→∞ 时 loss 的极限行为。

---

### §6.4 `dpo_training`（4 分，约 1 B200 hr）+ 三处评测

| 编号 | 任务 | 验收判据 |
|---|---|---|
| T12.1 | DPO 训练脚本 | 2 GPU（ref / policy 各一）、验证集约 200 条、梯度累积、RMSprop、有效 batch 64、β=0.1、lr 1e-6、1 epoch |
| T12.2 | 按验证准确率存最优 checkpoint | 验证指标 = chosen log-prob > rejected log-prob 的比例；交付 validation accuracy 曲线截图 |
| T12.3 | (b)(c) AlpacaEval + SST | 报 winrate / LC winrate / safe 比例，并与 SFT 模型对比 |
| T12.4 | (d) alignment tax | GSM8K + MMLU 上与 SFT 对比，2–3 句结论 |

**自查实验**
- 训练前先在验证集上算一次准确率，应接近 0.5。若一开始就 0.9+ → 先怀疑验证集与训练集重叠，或 log-prob 计算有误。
- 确认 π_ref 处于 eval 且不建梯度（显存与速度都不对时优先查这里）。
- 确认验证集的 200 条已从训练集切出。

**依赖提醒**：T12.4 需要 §3 与 §5 的数字做参照，所以 T12.4 必须排在 T1/T2 与 T8.1/T8.2 之后。

---

## 3. 依赖关系与建议执行顺序

```
T0.1（生成路径基建）
  ├─→ T1.* (§3.1 MMLU)   ─→ T8.1 (§5.1) ─┐
  ├─→ T2.* (§3.2 GSM8K)  ─→ T8.2 (§5.2) ─┤
  ├─→ T3.* (§3.3 Alpaca) ─→ T8.3 (§5.3) ─┼─→ T12.4 (alignment tax)
  └─→ T4.* (§3.4 SST)    ─→ T8.4 (§5.4) ─┤
                                          ├─→ T12.3 (DPO 评测)
T5.1 (look_at_sft) ─ 独立，随时可做      │
T6.* (data_loading) ✅ ─→ T7.* (SFT) ────┘─→ T9.1 (red teaming)
T10.1 (HH loader) ─→ T11.1 ✅ ─→ T12.1 ─→ T12.2 ─→ T12.3
```

**关键路径**：`T0.1 → T7（SFT 训练，最耗时） → T12（DPO 训练） → T12.3/T12.4（DPO 评测）`
**最大并行块**：§3 的四个 benchmark 彼此独立，可并行推进。

---

## 4. 三种投入模式（按可用时间选）

**模式 A：只做最省钱的部分（约 20% 工作量，0 GPU 训练小时）**
T5.1、T9.1、T10.1、T10.2 —— 两条纯文字题 + 一个纯 CPU 的数据加载题，共 10 分。外加 T1.1/T2.1/T6.*/T11.1 这些已通过的 adapter 对应的分值（约 7 分）。

**模式 B：做完 §3 与 §5 的评测链路（不需要训练，但需要推理机时）**
T0.1 + T1.*~T4.*（baseline 四件套）→ 此时 §5 无法做（还没有 SFT 模型）。适用于「想先掌握评测管线、暂不训练」。

**模式 C：完整做（约 20 B200 小时）**
按第 3 节顺序全做。注意 §4.2.2 标注 3 B200 hrs、§6.4 标注 1 B200 hr，加上 §3/§5 的推理开销。

---

## 5. 逐条验收用的命令汇总

```bash
# 三个必须通过的官方测试（已实测 7 passed）
.venv/bin/python -m pytest tests/test_metrics.py tests/test_data.py tests/test_dpo.py -q

# AlpacaEval 评测（PDF 原文）
alpaca_eval --model_outputs <preds.json> \
    --reference_outputs data/alpaca_eval/alpaca_eval_gpt4_turbo.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \
    --base-dir .

# SimpleSafetyTests 安全比例（PDF 原文）
uv run python scripts/evaluate_safety.py \
    --input-path <preds.jsonl> \
    --model-name-or-path /mnt/cs336-a5-supplement/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path <out.jsonl>
```

---

## 6. Writeup 骨架建议

把 PDF 里所有 `Deliverable:` 逐条抄成小标题，跑完一条填一条。当前 `writeup.md` 只有主 RL 作业的 §3–§7，没有任何 supplement 标题，这也是「看起来完全没做」的最直接原因。建议新增：

```
## Supplement §3 Zero-Shot Evaluation
### mmlu_baseline (a)(b)(c)(d)(e)(f)
### gsm8k_baseline (a)(b)(c)(d)(e)(f)
### alpaca_eval_baseline (a)(b)(c)(d)
### sst_baseline (a)(b)(c)(d)
## Supplement §4 Instruction Fine-Tuning
### look_at_sft
### data_loading (a)(b)
### sft_script / sft
## Supplement §5 Evaluating Our Instruction-Tuned Model
### mmlu_sft (a)(b)(c)  ### gsm8k_sft (a)(b)(c)
### alpaca_eval_sft (a)(b)(c)  ### sst_sft (a)(b)(c)
### red_teaming (a)(b)
## Supplement §6 Reinforcement Learning From Human Feedback
### look_at_hh (a)(b)
### dpo_loss
### dpo_training (a)(b)(c)(d)
```

---

## 7. 最高频翻车点 Top 5（写代码前先记住）

1. **解码参数串台**：§3/§5 要 greedy（temp 0.0），主 RL 作业链路默认 temp 1.0。改旧默认值会污染已有结果，应新增独立路径。
2. **提示词嵌套层次**：§3 用 system prompt，§5 用 Alpaca 模板；MMLU/GSM8K 还要先套一层 task prompt。层级错了性能会明显偏低。
3. **序列化格式**：AlpacaEval 要 JSON array，SST 要 JSONL；`dataset` 与 `prompts_final` 键名必须先查真实数据文件。
4. **解析器混用**：GSM8K 在 supplement（取最后数字）与主 RL 作业（取 `####` 后）的 parser 不同，PDF 专门加了 Note。
5. **梯度累积的除法位置**：loss 必须在 backward **之前**除以 k，否则有效 batch 的语义不对。
