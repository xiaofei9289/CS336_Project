# AutoDL 实例配置快照

记录时间：2026-09-16 22:35 +08:00。不含密码、密钥内容。实例关机或重装后 SSH 端口、容器名可能变化。

## 连接（本机 Mac）

- SSH 别名：`ssh autodl`
- 主机：`connect.westb.seetacloud.com`
- 用户：`root`
- 端口：**40193**（以 AutoDL 控制台「自定义服务」为准，重启后可能变）
- 密钥：`~/.ssh/id_ed25519`
- `~/.ssh/config`：`Host autodl`，`ProxyCommand none`，`IPQoS none`
- Clash Verge：**关掉 TUN** 再连；TUN 开着时 `seetacloud` 走 DIRECT，否则会进 fake-ip 超时

不要把实例密码写进仓库。

## 实例与路径

| 项 | 值 |
| --- | --- |
| 容器主机名 | `autodl-container-0k07c6dhtz-22e6acd7` |
| 作业目录 | `/root/Assignment02` |
| 数据盘 | `/root/autodl-tmp`（50G，nsys 报告放这里） |
| 系统盘 | overlay ~30G |
| nsys 报告 | `/root/autodl-tmp/nsys/*.nsys-rep` |
| nsys csv | `/root/autodl-tmp/nsys/stats/` |
| 本机备份 | `nsys_reports/`、`nsys_stats/`、`notes/nsys_profile_tables.md` |

容器内当时看到：`nproc`=208，内存约 1.0 Ti（多为宿主机视图）。AutoDL 控制台此前标称约 25 核 / 120 GB，写实验报告时以控制台规格 + `nvidia-smi` 为准。

## GPU / 驱动

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 1 |
| 显存 | 97887 MiB（约 96 GB） |
| UUID | `GPU-667a78c9-70c2-efcb-9852-7bda8564f51c` |
| 驱动 | 595.71.05 |
| Compute capability | 12.0 |
| CUDA toolkit 目录 | `/usr/local/cuda-12.8` |

## 软件栈（跑作业用这个，不要 `uv run`）

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Python | 3.12.3，`/root/miniconda3/bin/python` |
| Conda | `/root/miniconda3/bin/conda`（非交互 SSH 需把该 bin 加进 PATH） |
| PyTorch | 2.11.0+cu128 |
| `torch.version.cuda` | 12.8 |
| cuDNN | 9.19.0（`torch.backends.cudnn.version()` = 91900） |
| Triton | 3.6.0 |
| Nsight Systems CLI | 2026.5.1.161，包名 `nsight-systems-cli-2026.5.1`，命令 `/usr/local/bin/nsys` |

镜像原为 PyTorch 2.8 / CUDA 12.8 / Python 3.12，后升级 torch 到作业要求的 2.11.0+cu128。安装 torch 时不要开 `network_turbo`。

## 运行作业命令

```bash
cd /root/Assignment02
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/Assignment02/cs336-basics:/root/Assignment02
python -m cs336_systems.benchmark_lm --help
```

禁止：`uv run`（会按 lockfile 拉 CPU 版 torch）。

nsys 捕获测量区间（PyTorch NVTX 是未注册字符串，必须带环境变量）：

```bash
nsys profile \
  -o /root/autodl-tmp/nsys/<name> \
  --force-overwrite true \
  --trace cuda,nvtx \
  --capture-range nvtx \
  --nvtx-capture measurement \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  python -m cs336_systems.benchmark_lm --nvtx ...
```

## nsys_profile 网格（已跑完）

- medium：512 / 1024 / 2048
- large：256 / 512 / 1024（large@2048 探测 OOM）
- 本机数字底表：`notes/nsys_profile_tables.md`

## 本机同步过的内容

- 代码：Mac 仓库 `Assignment02` ↔ `/root/Assignment02`
- 当时远端 git：`916fe95`，相对 `origin/main` ahead 1；`benchmark.py` / `writeup_template.md` 有本地修改。不要往课程 stanford origin 推。
