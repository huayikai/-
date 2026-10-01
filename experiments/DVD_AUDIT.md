# DVD 两项实现修正的独立实验

这组代码只新增文件，不修改现有入口、注册表、learner、mixer 或原配置。服务器必须使用 `main_audit.py`；原 `main.py` 不会注册新 learner。下面命令假定从服务器仓库根目录运行，代码位于 `src/`；若仓库根目录直接包含 `main.py`，去掉命令中的 `src/`。

## 两项修正

`audit_fix_td_lambda=True`：按已经对齐的下一时刻 SARSA 联合 Q 值递推，包括最后一个有效 transition 的奖励。真正终止时目标为该步奖励；时间限制或序列截断时目标为该步奖励加下一状态的一步 bootstrap。每条轨迹按自己的有效 mask 截止，padding 目标置零。递推是

```text
G[t] = r[t] + gamma * (1 - terminated[t])
                 * ((1 - lambda) * Q_next[t] + lambda * continuation[t])
continuation[t] = G[t+1]，若下一步有效；否则为 Q_next[t]
```

`audit_isolate_dvd_aux_hidden=True`：仅对 counterfactual router 的 DVD 辅助损失，使用 detached hidden-state 重算 DVD 权重，阻止辅助损失更新 agent 的 RNN，同时保留 GAT 和 DVD hypernetwork 的梯度。mixed 主损失仍保留 agent Q 和 hidden-state 的梯度。这不是全局 detach；基础 DVD 没有这个独立辅助损失，因此不开此项。开启后在线 mixer 会多一次 GAT 前向，可能增加训练耗时；`no_grad` 的目标前向不重算。

新 learner 是当前 `dvd_nq_learner_with_sarsa.py` 的独立副本，原 TD helper 被保留用于关闭开关的对照。其余优化器、RND、target 更新、路由标签和损失系数沿用现有实现。新入口沿用原训练流程，仅在本进程注册新 learner。

## 配置和对照

| 配置 | 结构与用途 | 默认修正 |
| --- | --- | --- |
| `dvd_audit_router` | 当前 safe-v2 counterfactual router，epsilon_start=0.75 | 两项都开启 |
| `dvd_audit_base` | 之前的基础 DVD 配置，epsilon_start=1.0 | 仅 TD |
| `dvd_audit_bm` | `dvd_takeover` 的纯 BM warm-up 对照，epsilon_start=0.75 | 仅 TD |

BM 对照显式使用 `takeover_bm_abs=False`，并把 warm-up 设为 20M，大于默认训练的 10.05M，因此训练期间不接管 DVD。若改动训练预算，应保证整个训练期间小于 warm-up；该配置不是永久关闭 takeover。

router 的 2×2 实验为：

| 实验名中的 variant | TD 修正 | 辅助 hidden 梯度隔离 |
| --- | --- | --- |
| `legacy` | False | False |
| `td` | True | False |
| `aux` | False | True |
| `both` | True | True |

`legacy` 是用新入口复现当前行为的必要对照。基础 DVD 与 BM 只运行 `legacy`、`td`。router 的 TD 修正版应与同样开启 TD 修正的 BM 比较；基础 DVD 与 BM 比较前还要匹配 epsilon_start 等训练配置，不能直接把默认基础配置和默认 BM 配置的差值归因于 DVD。

## 先检查，再运行

本地 CPU 检查不需要 SMAC、SC2 或 Sacred，但需要已有的 torch、numpy、PyYAML：

```bash
python3 src/_test_dvd_audit.py
python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 --audit-print-config with audit_fix_td_lambda=False
```

12 个测试覆盖末尾奖励、终止/截断/padding、独立 forward-view λ-return 对照、实际 GRU 的辅助/主损失梯度路径、前向值完全一致、关闭两项后含 RND 的单步更新与旧版逐参数一致、各配置和入口注册。测试里的 EpisodeBatch/MAC 输入及日志对象使用小型替身；本机缺少 SMAC/Sacred，尚未在真实 SC2 环境启动训练。服务器应先执行测试与一次短启动，确认环境可用。

队列脚本默认只打印命令；加 `--run` 才启动，并要求显式指定 GPU。每张 GPU 同时只启动一个任务。

```bash
# 先查看 seed 79 的四组命令
python3 src/experiments/run_dvd_audit.py --method router --seeds 79 --gpus 0,1,2,3

# 可选：10000 步仅检查启动流程，不用于判断算法效果
python3 src/experiments/run_dvd_audit.py --method router --variants legacy both --seeds 79 --gpus 0,1 --t-max 10000 --run

# 正式运行四组，各自沿用 10.05M 预算
python3 src/experiments/run_dvd_audit.py --method router --seeds 79 --gpus 0,1,2,3 --run

# BM 和基础 DVD 的 TD 开/关对照，根据首轮结果选择运行
python3 src/experiments/run_dvd_audit.py --method bm --seeds 79 --gpus 0,1 --run
python3 src/experiments/run_dvd_audit.py --method base --seeds 79 --gpus 0,1 --run
```

短启动和正式训练使用相同实验名，但 Sacred 每次创建独立 run；TensorBoard 沿用原流程的时间戳子目录，分析时按实际预算区分。地图可用 `--map 6h_vs_8z` 覆盖。若某个任务返回非零，队列停止提交剩余任务，已启动任务继续到结束。

单独运行并自定义超参数的示例：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=79 use_tensorboard=True name=audit_router_both_3s5z_vs_3s6z_s79 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True
```

## 如何判断是否值得继续

修正实现不保证胜率提高。先用 seed 79 的四组辨别是哪项改变影响训练；一颗种子只能做排错和资源决策。若准备形成结论，应重复相同预算、配对种子（包括原 seed 82），同时报告成功与失败运行；正式验证再用未参与选择的新种子。

至少比较相同区间的测试胜率 AUC、9–10M 尾段胜率、非有限值和中断情况。router 还应检查 gate 与 route target 的相关性、gate 方差、DVD 相对 BM 的 TD 优势。不要只选择最高点，也不要把 5M 中断运行当作完整 10M 结果。若修正后仍不能稳定优于同配置 BM，停止继续扫 gate 超参更合理；基础 DVD 的 TD 开/关对照可以帮助判断之前较好结果是否依赖旧目标计算。

每次新入口运行会把关键源码、实际使用的 YAML 配置加入 Sacred 源码保存列表，并在配置中记录 `audit_source_manifest` 的 SHA-256；CLI 覆盖参数由 Sacred 记录。结果仍存储到原流程的 `ablation_results`。原 mixer 是共享依赖，比较这轮四组期间不要再改其实现；manifest 用来核对是否跑了同一套源码。

## 提交文件

只需把以下新增文件提交到 Git。之前日志分析产生的 `analysis/` 含较大原始聚合 JSON，不是训练依赖，不必随这组代码提交。

```text
main_audit.py
_test_dvd_audit.py
learners/dvd_audit_learner.py
utils/td_lambda_audit.py
modules/mixers/dvd_counterfactual_router_audit.py
config/algs/dvd_audit_router.yaml
config/algs/dvd_audit_base.yaml
config/algs/dvd_audit_bm.yaml
experiments/run_dvd_audit.py
experiments/DVD_AUDIT.md
```
