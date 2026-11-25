# Google Research Football：BACM 首批实验

## 2026-11-23：第四张卡扩展 counterattack_easy

BM、QMIX、VDN三条3vs1对照继续，GPU3用于新增地图的完整BACM性能实验。
采用3vs1当前选定的λ=1.0、lr=0.0005、50万步探索退火，固定seed80，
从头训练10.05M、100局独立贪心评估。此配方尚未在counterattack_easy调优。
预设自动选择academy_counterattack_easy与4个受控球员。

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=counterattack_easy with seed=80 \
  name=bacm_grf_countereasy_s80_mappo100_10m_lam10_lr0005 \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 \
  t_max=10050000 test_nepisode=100 use_rnd=True
```

输出仍为ablation_results_football。保留RND原有约6M衰减关闭日程，
其他共同训练参数保持当前足球配置。旧反击实验为旧种子/评估协议及不同超参数，
不与本条作为同协议配对比较。先看完整方法能否学会，再补本地图相同seed的基线。

## 2026-11-23：首批足球基线（BM、QMIX、VDN）

本批先用三张卡跑已接入足球入口的纯BM、QMIX、VDN。固定3vs1、seed80、
TD(λ)=1.0、lr=0.0005、50万步ε退火和10.05M预算，每次100局贪心评估，
训练与测试环境独立。原服务器代码即可执行，不使用检查点，从头训练。

BACM配对参照为 `bacm_grf_3v1_s80_mappo100_10m_eps500k`，
4–5M平均测试成功率67.56%，9–10M为51.08%。比较双方同一训练窗口。
BM共用BACM的learner与RND日程（约6M关闭），20M接管预热使本预算内保持纯BM。
QMIX/VDN使用原nq_learner与Q-learning目标、关闭RND。外部比较按完整配方解释。
该批采用共同训练参数，并非分别调优后的原论文结果。

在PyMARL2根目录激活pymarl后，三个终端各执行一条。

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_football.py --debug --capture=no \
  --config=bm --scenario=3_vs_1 with seed=80 \
  name=baseline_bm_grf_3v1_s80_mappo100_10m \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 \
  t_max=10050000 test_nepisode=100 use_rnd=True
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_football.py --debug --capture=no \
  --config=qmix --scenario=3_vs_1 with seed=80 \
  name=baseline_qmix_grf_3v1_s80_mappo100_10m \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 \
  t_max=10050000 test_nepisode=100 use_rnd=False
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_football.py --debug --capture=no \
  --config=vdn --scenario=3_vs_1 with seed=80 \
  name=baseline_vdn_grf_3v1_s80_mappo100_10m \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 \
  t_max=10050000 test_nepisode=100 use_rnd=False
```

三条仍输出至 `ablation_results_football`，每条4个训练环境与4个测试环境，
batch64、buffer2000、RNN64、gamma0.99。日志关注训练成功率、测试成功率、
test_n_episodes与test_won_count。本批未启动服务器训练。
DVD尚未接入本入口，明确版本及验证后再给指令。

## 2026-11-20：MAPPO 风格评估协议

当前协议为 `env_args.engine_seed_mode=mappo`、`football_eval_separate_envs=True`。
环境初始化时设置一次 Python random / NumPy 随机种子，不向 GRF 传固定的
`game_engine_random_seed`，由 GRF 在每个新回合自然生成引擎种子。

训练 worker 使用 `seed + rank * 1000`，独立评估 worker 使用
`seed * 50000 + rank * 10000`。训练与测试各有 4 个独立环境进程。
评估沿用贪心动作，每 1 万训练环境步累计 100 局后统计一次，不重置评估随机序列。
只对足球启用独立评估进程，原 SMAC 设置不变。

日志新增 `test_n_episodes`（正式配置应为 100）和 `test_won_count`；
`test_battle_won_mean = test_won_count / test_n_episodes`。
100 局提高估计分辨率，并不保证学习曲线平滑。

旧显式固定引擎种子的运行另列为历史。旧四条数据全部 1,732 个成功率点为
25% 的倍数，与重复随机条件一致；尚未用冻结策略逐轨迹实测重复次数。
本轮既改变训练随机条件，也改变测试协议，应以新名称从头训练默认参照，
不能沿用旧默认运行当作新版调参的配对参照。

服务器同步包 `server_patch_mappo_eval.zip` 内按 `src/` 结构放置文件。
从 `/home/zhangbei/pymarl2` 解压同步后，先按下文检查原生环境和短训练。
本地 34 项检查中 33 通过、1 项缺 Sacred 跳过。原生 GRF/服务器验证仍待执行。

参考：
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/envs/football/Football_Env.py
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train/train_football.py
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train_football_scripts/train_football_3v1.sh

## 同步与安装

本地 `D:/论文/代码/` 对应服务器 `/home/zhangbei/pymarl2/src/`。
把本轮文件按相对目录同步到 `src/`，保留服务器仓库根目录的安装脚本。

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
# 仅在未安装足球时执行；已有可用安装无需重装。
bash install_gfootball.sh
```

上游安装脚本使用 `gfootball==2.10.2 gym==0.11`，并安装 SDL/Boost 等系统库，
可能需要管理员权限。沿用已有 PyMARL2/Python 环境；本轮未远程执行安装，
不自动升级 Gym、NumPy 或 PyTorch。若安装报错，保留完整输出再适配。

上游参考：
- https://github.com/hijkzzz/pymarl2/blob/master/install_gfootball.sh
- https://github.com/hijkzzz/pymarl2#google-football
- https://github.com/google-research/football/blob/master/gfootball/doc/observation.md

## 先做真实环境检查

```bash
python3 src/experiments/2026-11-19_GFOOTBALL/smoke_env.py --scenario=3_vs_1
python3 src/experiments/2026-11-19_GFOOTBALL/smoke_env.py --scenario=counterattack_easy
```

检查环境安装、reset/step、115 维观测、19 动作、奖励与终止。
随机策略没有进球是正常现象；这不是训练性能实验。

再用一次短训练确认多进程、回放、BACM 更新、模型与指标保存：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=3_vs_1 with seed=80 \
  name=smoke_bacm_3v1_s80_mappo100 t_max=10000 batch_size=4 buffer_size=16 \
  test_nepisode=100 test_interval=5000 local_results_path=football_smoke
```

这条为调试配置，不能混进正式实验汇总。`t_max` 按完整并行回合推进，可能小幅超出。

## 第一批：四张卡全部跑 BACM

| GPU | 场景 | 智能体数 | seed |
| --- | --- | ---: | ---: |
| 0 | academy_3_vs_1_with_keeper | 3 | 79 |
| 1 | academy_3_vs_1_with_keeper | 3 | 80 |
| 2 | academy_counterattack_easy | 4 | 79 |
| 3 | academy_counterattack_easy | 4 | 80 |

每条 505 万环境步，4 个并行环境，100 局评估，每 1 万步测试，保存至
`/home/zhangbei/pymarl2/ablation_results_football`。模型约每 100 万步保存。
先验证 BACM，后续再决定扩种子和性能对照，不新增组件消融。

四个终端各执行一条：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=3_vs_1 with seed=79 name=bacm_grf_3v1_s79_mappo100
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=3_vs_1 with seed=80 name=bacm_grf_3v1_s80_mappo100
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=counterattack_easy with seed=79 name=bacm_grf_countereasy_s79_mappo100
```

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=counterattack_easy with seed=80 name=bacm_grf_countereasy_s80_mappo100
```

## 本轮定义和后续对照

- 观测为 `simple115v2`；集中训练状态拼接所有受控智能体观测。
  这是拼接观测，不是环境提供的额外全局状态。输入含双方位置、方向和球信息，
  不能直接宣称复现 SMAC 的局部可观测条件。
- 训练外部奖励采用 `scoring,checkpoints`，取受控球员奖励的均值；避免 GRF
  重复发送团队奖励后被智能体数量放大。旧 `gfootball.yaml` 的默认 sum 保留。
- 成功率 `test_battle_won_mean` 使用引擎未塑形的 `score_reward` 判定进球，
  不用正的 checkpoints 奖励判断成功。仅适用于本轮单次进攻即结束的 Academy 场景。
- `test_return_mean` 为足球外部塑形回报；BACM/BM 保留 RND 训练探索，
  RND 不进入这个评估回报。`test_score_reward_mean` 为每回合进球净得分。
- 使用 detached 候选隐藏路径；其余 BACM 结构及损失保持现有定义。
- 学习率 0.001、TD(λ)=0.3、gamma=0.99、RNN=64、初始探索率 1.0、
  50 万步退火到 0.05。属于足球开发起始配方，尚无性能验证。
- 训练与评估种子按顶部 MAPPO 公式分配。seed 为整次训练的配置编号，
  多个 worker 不计作独立训练重复。

入口同时支持 `--config=bm`、`--config=qmix`、`--config=vdn`。
BM 与 BACM 共用 RND 和训练目标配方；QMIX/VDN 使用原 `nq_learner`、
原生 Q-learning 目标并关闭 RND，所以外部比较属于完整训练配方的比较。
暂未开放 Qatten：其本地实现使用 SC2 专用状态切片，不能直接改环境名套用。
QPLEX/QTRAN 不在这个足球入口的已验证范围内。

其他场景预设：`counterattack_hard`（4 agents）、`pass_and_shoot`（2 agents）。
使用 `--scenario` 选择，入口自动匹配地图与智能体数，不单独覆盖 map_name/num_agents。
各预设上限 400 步；区分 wrapper 强制截断与环境自然结束。

配置预览无需安装足球引擎：

```bash
python3 src/main_football.py --football-print-config --config=bacm --scenario=3_vs_1 with seed=79
```

回传 `ablation_results_football` 的 Sacred 与 TensorBoard 文件，保留所有运行。
主看完整曲线及 400–500 万步平均成功率；模型文件可单独存放以减少回传体积。
未来按成功案例选择补基线时，图表注明案例选取方式，原始失败运行继续保留。

## 当前完成范围

已完成本地接口检查和真实 BACM 网络的合成足球形状更新检查。
最新本地运行 34 项检查，33 项通过、1 项缺少 Sacred 跳过，零失败；详见 `validation.json`。
本机无 GRF 引擎，未验证服务器原生引擎及真实进球表现；上述 smoke 命令用于补齐这一步。
未启动服务器训练，未自动提交或推送。

## 2026-11-25：三个 Academy 场景，固定 seed=80，8M 摸底

沿用 `bacm_grf_3v1_s80_mappo100_10m_eps500k__2026-11-22_12-28-21`
的训练参数：TD(lambda)=1.0、lr=0.0005、探索退火500k、RNN=64。
先分别从头训练一个种子；主看7–8M的平均测试成功率和完整曲线。
`t_max=8050000` 留少量余量确保8M附近有评估记录，汇总只取不超过8M的点。
RND仍按原6M周期衰减，不随训练预算改变。

新增 `corner` 预设：`academy_corner`，10个受控智能体。
现有环境封装和BACM按智能体数自动构建观测、状态、网络，无需改变算法。
MAPPO对应脚本也使用10个受控智能体；GRF场景定义原生时长为400。
参考：
- https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train_football_scripts/train_football_corner.sh
- https://github.com/google-research/football/blob/master/gfootball/scenarios/academy_corner.py

先同步本地 `config/football.yaml` 到服务器 `src/config/football.yaml`。
也可将 `server_patch_corner.zip` 放到服务器PyMARL2根目录后解压：

```bash
cd ~/pymarl2
conda activate pymarl
unzip -o server_patch_corner.zip
for scenario in pass_and_shoot counterattack_hard corner; do
  python3 src/experiments/2026-11-19_GFOOTBALL/smoke_env.py --scenario="$scenario" || break
done
```

三个场景都通过原生环境检查后，分别在三个终端运行：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=pass_and_shoot with seed=80 \
  name=bacm_grf_passshoot_s80_mappo100_8m_eps500k \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 rnn_hidden_dim=64 \
  t_max=8050000 batch_size_run=4 test_nepisode=100 test_interval=10000 \
  use_rnd=True local_results_path=ablation_results_football
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=counterattack_hard with seed=80 \
  name=bacm_grf_counterhard_s80_mappo100_8m_eps500k \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 rnn_hidden_dim=64 \
  t_max=8050000 batch_size_run=4 test_nepisode=100 test_interval=10000 \
  use_rnd=True local_results_path=ablation_results_football
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_football.py --debug --capture=no \
  --config=bacm --scenario=corner with seed=80 \
  name=bacm_grf_corner_s80_mappo100_8m_eps500k \
  td_lambda=1.0 lr=0.0005 epsilon_anneal_time=500000 rnn_hidden_dim=64 \
  t_max=8050000 batch_size_run=4 test_nepisode=100 test_interval=10000 \
  use_rnd=True local_results_path=ablation_results_football
```

每条命令占用4个训练环境进程和4个独立评估环境进程。
检查预期为：pass_and_shoot的n_agents=2/state_shape=230，
counterattack_hard的4/460，corner的10/1150；均为obs_shape=115、n_actions=19。
本地合成数据训练检查不替代服务器GRF原生引擎检查。
