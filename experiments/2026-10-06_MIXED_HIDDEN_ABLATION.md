# 2026-10-06：mixed hidden 梯度通路消融（待运行）

根据10月5日下午0.05四组结果安排本轮：固定mixed权重0.25，只切断mixed损失经DVD权重返回agent hidden的直接通路。agent Q路径以及GAT/DVD参数仍正常训练。保留正确TD、辅助hidden隔离、BM锚定目标、损失归一化、学习率、探索与RND时钟。尚未启动服务器训练。

`audit_detach_mixed_hidden=False`为默认值，保留原audit训练行为；本轮命令显式设为True。它与`audit_isolate_dvd_aux_hidden`独立。关闭辅助隔离时，辅助损失仍可返回hidden，不受mixed开关影响。普通router文件和旧learner未修改；新实验使用`main_audit.py`。

四卡分别为3s79、3s81、6h79、6h80。`t_max=5050000`用于覆盖5M评估，指标仍截取0–5M AUC与4–5M平均胜率；已有同种子BM/Router 0.25/0.05作参照。5M不能检验3s81已知的9–10M回退。

命令见[四卡命令文件](2026-10-06_mixed_hidden_commands.sh)，每条在一个终端单独运行。先把更新后的`D:\src`同步到服务器`/home/zhangbei/pymarl2/src`，再从`~/pymarl2`运行。入口、保存路径helper、默认配置、runner及本轮audit mixer/learner/router配置均需使用更新文件。结果固定保存到`/home/zhangbei/pymarl2/ablation_results_10_6/`。

`audit_log_agent_gradients=True`按`learner_log_interval`周期记录同批次、全局裁剪前agent参数上的BM/mixed原始梯度范数、夹角及带损失系数的范数比。`autograd.grad`不累加`.grad`、不更新参数；它会增加少量诊断计算。夹角或范数比不能单独证明性能因果，也不是Adam实际更新比例；零范数时夹角约定为0。

记录标签为`audit_agent_bm_grad_norm`、`audit_agent_mix_grad_norm`、`audit_agent_bm_mix_grad_cosine`、`audit_agent_weighted_mix_bm_grad_ratio`以及开关状态`audit_detach_mixed_hidden`。源码已在现有audit manifest中记录。

CPU验证入口：`python3 src/_test_mixed_hidden_audit.py`、`python3 src/_test_dvd_audit.py`。验证默认行为、前向值、零hidden回退、两开关独立性、Q和GAT/DVD梯度保留、agent更新，以及诊断开关不改变单步参数更新。实际SC2效果需四组训练后比较。
