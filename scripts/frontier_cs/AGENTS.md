# FrontierCS Agent Quick Start

这份文件是给每次接手 `scripts/frontier_cs/` 的模型看的启动指南。先按这里扫一遍，再去改脚本或发起 run。

## 入口

- 首选启动入口是 `scripts/frontier_cs/launch.sh`。
- 具体命令例子参考 `scripts/frontier_cs/cheatsheet.sh`。
- `launch.sh` 会串起三件事：
  1. 生成 `data/frontiercs/{train,val}.jsonl`
  2. 启动 Frontier-CS judge，默认 `http://localhost:8082`
  3. 下载缺失的 HF 模型权重到 `models/<MODEL_NAME>`
  4. 转换缺失的 `models/<MODEL_NAME>_torch_dist`
  5. 调用 `scripts/launch.sh` 启动 SLIME training

常用 debug 模板：

```bash
RUN_NAME=qwen3.5-35B-A3B-frontiercs-debug-n32-use-tis

bash scripts/frontier_cs/launch.sh \
  --script scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B-debug.sh \
  --run-name "${RUN_NAME}" \
  --skip-data \
  --skip-judge | tee "logs/frontiercs/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log"
```

如果首次运行或不确定数据 / judge / 模型状态，不要加 `--skip-data` / `--skip-judge` / `--skip-model`。如果确认 `data/frontiercs` 已经准备好、judge health check 正常、HF 和 torch_dist 权重都存在，可以跳过它们节省时间。

## 运行前检查

- 当前目录通常应在 repo root：`/fsx-alignment/home/zhuofeng/slime`。
- 确认模型权重和 torch_dist 权重存在。各 run script 顶部的 `Prerequisites` 写了对应路径。
- `launch.sh` 默认会补齐缺失的模型权重和 torch_dist 权重；如果手动使用底层 run script，仍需自己确认路径。
- 如果没有 `WANDB_KEY`，`launch.sh` 会自动设置 `WANDB_MODE=offline`，避免新机器首次运行时因为 wandb 登录失败中断。
- 如果手动查 judge：`curl -sf http://localhost:8082/health`。
- 日志统一放到 `logs/frontiercs/`，不要只在终端里裸跑。

## 运行监控

- 启动 run 时必须用 `tee` 写入一个带 `RUN_NAME` 和时间戳的日志文件，例如 `logs/frontiercs/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log`。
- 如果模型发起了长时间运行命令，必须在后续持续监控对应日志输出，而不是只等待进程结束。常用方式：

```bash
tail -n 200 -f "logs/frontiercs/${RUN_NAME}-YYYYMMDD-HHMMSS.log"
```

- 如果不确定当前 run 对应哪个日志，先按时间倒序找最近文件：

```bash
ls -lt logs/frontiercs | head
```

- 监控时重点看：judge health check、SGLang worker 是否 unhealthy / OOM、rollout 进度、训练 step、wandb 链接、traceback，以及训练 / eval 指标里的 `reward` 和 `max_score`。发现异常要先记录对应日志文件、step、关键指标变化和关键报错，再决定是否停 run 或调整参数重试。

## Run 命名规范

- 任何和默认参数不同的 run，都必须把差异写进 `RUN_NAME`，方便之后区分 wandb、checkpoint、rollout dump 和日志。
- `RUN_NAME` 应包含：模型、任务、debug/full、关键参数变化。
- 例子：
  - `qwen3.5-35B-A3B-frontiercs-debug-n32-use-tis`
  - `qwen3.5-27B-frontiercs-debug-n32`
  - `qwen3.6-35B-A3B-frontiercs-lr5e-8-mlp-only`

## 调试和调参约定

- 调参过程中默认不需要 save model。只有确实要保留 checkpoint 时才设置 `SAVE_INTERVAL` 或确认脚本里有 `--save`。
- Debug run 优先使用 `*-debug.sh` 脚本，不要直接改 full run 的参数来做短实验。
- 运行 Qwen3.5-9B 时，先使用 `scripts/frontier_cs/run-frontiercs-qwen3.5-9B-debug.sh` 的默认参数；如果用户要求改参数，必须通过环境变量 / `launch.sh` 参数透传，并且在 `RUN_NAME` 里明确体现修改，不能直接改脚本默认值。
- FrontierCS 的 `ROLLOUT_MAX_RESPONSE_LEN` / max response 默认必须保持 `81920`；除非用户单独明确指定，否则一定不能改小。
- 不要对默认采样温度做任何修改；除非用户单独明确指定，否则保持脚本里的默认 temperature。
- 如果需要调参，必须通过传参 / 环境变量覆盖的方式把参数传进去，不要直接修改 `*-debug.sh` 里的默认配置。只有当脚本缺少必要可调入口时，才新增环境变量默认值，并同步更新本文件。
- 如果需要避免 all-zero / zero-std rollout 进入训练，优先通过 `DYNAMIC_SAMPLING_FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` 透传开启 dynamic sampling，并在 `RUN_NAME` 里标明 `nonzero-std-filter`；不要为了这个直接改采样温度或 max response。
- 9B debug 脚本支持通过 `SGLANG_MEM_FRACTION_STATIC` 临时透传 `--sglang-mem-fraction-static`；只有遇到 SGLang OOM / worker unhealthy 时才传，并且 `RUN_NAME` 必须写明 `memfrac`。
- Ray 刚启动后 dashboard agent 可能还没 ready；提交 job 前应等 `http://127.0.0.1:8265/api/jobs/` 可访问，避免 `No available agent to submit job`。
- 对 27B debug 脚本，常用覆盖变量包括：
  - `NUM_ROLLOUT`
  - `ROLLOUT_BATCH_SIZE`
  - `N_SAMPLES_PER_PROMPT`
  - `ROLLOUT_MAX_RESPONSE_LEN`
  - `GLOBAL_BATCH_SIZE`
  - `DYNAMIC_SAMPLING_FILTER_PATH`
  - `OVER_SAMPLING_BATCH_SIZE`
  - `PARTIAL_ROLLOUT`
  - `SAVE_INTERVAL`
  - `LOAD_DEBUG_ROLLOUT_DATA`
  - `LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE`
  - `DEBUG_TRAIN_ONLY`
- 如果需要复用 rollout dump，优先通过 `LOAD_DEBUG_ROLLOUT_DATA` / `DEBUG_TRAIN_ONLY=1` 走脚本已有开关。

## 数据和 Judge

- 数据生成脚本是 `scripts/frontier_cs/prepare_frontiercs_jsonl.py`。
- 默认输出目录是 `data/frontiercs`。
- `--full-for-both` 会把全部题目同时写入 train 和 val，当前 run script 默认按这个模式消费。
- judge 代码在 `FrontierSmith/Frontier-CS/algorithmic`，由 `launch.sh` 构建 / 启动 Docker image `frontiercs-judge`。
- 模型下载 / 转换由 `launch.sh` 根据 `--model` 自动推导：`MODEL_NAME`、`HF_REPO`、`CONVERT_SCRIPT`。自定义 `--script` 时要么设置这三个环境变量，要么显式加 `--skip-model`。

## 修改脚本时

- 优先保留现有结构：`CKPT_ARGS`、`ROLLOUT_ARGS`、`EVAL_ARGS`、`PERF_ARGS`、`GRPO_ARGS`、`OPTIMIZER_ARGS`、`WANDB_ARGS`、`SGLANG_ARGS`、`MISC_ARGS`。
- 新增可调参数时，优先用环境变量加默认值，而不是把临时参数硬编码死。
- 如果新增了一个重要开关，也把它写回本文件和 `cheatsheet.sh` 的示例里。
- 改动后至少做 shell 语法检查：

```bash
bash -n scripts/frontier_cs/launch.sh
bash -n scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B-debug.sh
bash -n scripts/frontier_cs/run-frontiercs-qwen3.5-27B-debug.sh
```

## 实验记录（exp.csv）

每发起一次实验（无论成功还是失败），都必须把这次 run 追加记录到 `scripts/frontier_cs/exp.csv`，方便后续复盘和对比。不要等 run 结束才补记，启动后就先写一行，run 结束后再回填结果。

`exp.csv` 是标准 CSV，表头固定为四列，按顺序为：

1. `ExperienceName`：本次实验的名字，直接用 `RUN_NAME`，保证和 wandb / checkpoint / 日志一致。
2. `ModelSize`：使用的模型规模，例如 `35B-A3B`、`27B`、`9B`。
3. `Commands`：本次实验具体使用的启动命令（含关键的环境变量覆盖和 `launch.sh` 参数）。命令里有逗号或换行时，整列用双引号包起来，内部双引号写成 `""`。
4. `FailReason`：如果失败，写明 fail 的原因（关键报错、step、异常指标等）；成功则留空或写 `-`。

追加示例：

```bash
echo 'qwen3.5-9B-frontiercs-debug-n32,9B,"bash scripts/frontier_cs/launch.sh --script scripts/frontier_cs/run-frontiercs-qwen3.5-9B-debug.sh --run-name qwen3.5-9B-frontiercs-debug-n32 --skip-data --skip-judge",-' \
  >> scripts/frontier_cs/exp.csv
```

写完后确认这一行的列数和表头一致，避免破坏 CSV 格式。

## 经验回写

调试 FrontierCS 相关 script 时，如果发现新的 best practice、坑点、参数组合或恢复方式，直接补充到当前 `scripts/frontier_cs/AGENTS.md`。这个文件的目标是让下一次模型能快速、安全地启动 FrontierCS run。
