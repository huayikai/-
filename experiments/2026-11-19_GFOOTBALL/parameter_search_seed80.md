# 固定 seed 80 调参：MAPPO 风格测试协议

2026-11-20 更新。先同步 server_patch_mappo_eval.zip 内的新版文件。
训练与测试进程分开，GRF 自然生成逐回合引擎种子，每次贪心评估 100 局。
旧固定引擎种子的 seed80 不再作为新版调参参照，先从头跑新版默认参照。

固定 academy_3_vs_1_with_keeper、seed80，每条505万训练环境步，
4个训练环境和4个测试环境。除指定候选项外，batch64、buffer2000、RNN64、
gamma0.99、lr0.001、TD(lambda)=0.3、初始探索率1.0、50万步退火、RND开启。

| GPU | 设置 | 相对新版默认参照的变化 |
| --- | --- | --- |
| 0 | 新版默认参照 | 无 |
| 1 | 降低学习率 | lr=0.0005 |
| 2 | 改 TD(lambda) | td_lambda=1.0 |
| 3 | 关闭 RND | use_rnd=False |

在PyMARL2根目录激活pymarl后，四个终端各执行一条。

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_football.py --debug --capture=no --config=bacm --scenario=3_vs_1 with seed=80 name=bacm_grf_3v1_s80_mappo100_base
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_football.py --debug --capture=no --config=bacm --scenario=3_vs_1 with seed=80 name=bacm_grf_3v1_s80_mappo100_lr0005 lr=0.0005
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_football.py --debug --capture=no --config=bacm --scenario=3_vs_1 with seed=80 name=bacm_grf_3v1_s80_mappo100_lam10 td_lambda=1.0
```

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_football.py --debug --capture=no --config=bacm --scenario=3_vs_1 with seed=80 name=bacm_grf_3v1_s80_mappo100_nornd use_rnd=False
```

RNN128留下一批。比较完整曲线及400–500万步平均成功率，不按单次峰值选参数。
正式评估应记录test_n_episodes=100，成功率为test_won_count/100。
100局测试比原32局耗时更多，测试步数不计入训练t_env。
先执行commands.md中的新版原生环境检查和短训练，再启动正式调参。
固定seed用于开发；参数确定后再扩种子、补基线。旧协议原始数据保留。
