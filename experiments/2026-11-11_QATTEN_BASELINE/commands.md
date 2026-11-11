# 独立 Qatten：两批八组固定条件

沿用 QPLEX、QTRAN-base 的地图和种子。每组训练 5,050,000 环境步，约每 10,000 步评估一次，每次 32 局。使用 [PyMARL2 的加权四头 Qatten 配方](https://github.com/hijkzzz/pymarl2/blob/master/src/config/algs/qatten.yaml)：`n_mac` / `n_rnn` / `nq_learner` / `qatten`，Adam 0.001，TD(λ)=0.6，8 并行环境，batch 128，buffer 5000。比较各算法配方；结构归因仍看内部对照。

Qatten 使用全局状态开头的友方单位特征，现有 SC2 路径会按地图计算特征宽度。训练入口记录实际 learner、mixer、控制器、agent 和环境文件的指纹。只将 NQLearner 的梯度范数转成标量日志，矩阵游戏调试导入推迟到其专用分支；训练目标不变。

服务器先同步本次改动的 `src/main_baseline.py`、`src/learners/nq_learner.py`、`src/_test_qatten_baseline.py`。同一服务器应已有 `src/_test_external_baseline.py`；可与 Qatten 配置、mixer、控制器和 agent 一起从同一提交同步。Windows 本地共通过九项 CPU 检查（Qatten 三项，QPLEX 和 QTRAN 各三项）；尚未在服务器 SC2/CUDA 上跑。

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
python3 src/_test_qatten_baseline.py
```

建议正式训练前做两条 25,000 步的短预检，输出另存。确认两张图都能评估并记录 `loss_td`，然后开四个终端，各自进入项目并激活环境。若预检报错，先保留完整报错。

```bash
for map in 3s5z_vs_3s6z 6h_vs_8z; do
  CUDA_VISIBLE_DEVICES=0 python3 src/main_baseline.py --debug --capture=no \
    --config=qatten --env-config=sc2 with env_args.map_name="$map" seed=84 \
    t_max=25000 use_tensorboard=True local_results_path=qatten_smoke \
    name="smoke_qatten_$map" || break
done
```

## 第一批（四个终端各运行一条）

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=84 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_3s5z_vs_3s6z_s84
CUDA_VISIBLE_DEVICES=1 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=85 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_3s5z_vs_3s6z_s85
CUDA_VISIBLE_DEVICES=2 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=6h_vs_8z seed=92 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_6h_vs_8z_s92
CUDA_VISIBLE_DEVICES=3 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=6h_vs_8z seed=93 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_6h_vs_8z_s93
```

## 第二批（第一批释放对应 GPU 后运行）

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=86 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_3s5z_vs_3s6z_s86
CUDA_VISIBLE_DEVICES=1 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=87 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_3s5z_vs_3s6z_s87
CUDA_VISIBLE_DEVICES=2 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=6h_vs_8z seed=94 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_6h_vs_8z_s94
CUDA_VISIBLE_DEVICES=3 python3 src/main_baseline.py --debug --capture=no --config=qatten --env-config=sc2 with env_args.map_name=6h_vs_8z seed=95 t_max=5050000 use_tensorboard=True local_results_path=ablation_results_11_6 name=baseline_qatten_pymarl2_5m_6h_vs_8z_s95
```

回传 `ablation_results_11_6` 中对应的 `sacred` 和 `tb_logs`。短预检不计入正式实验。保留全部八组，比较距 500 万步最近的原始胜率、400–500 万步均值及 505 万步内的末五次均值。
