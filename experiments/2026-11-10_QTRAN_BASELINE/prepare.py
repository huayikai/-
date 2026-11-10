"""Lock QTRAN-base conditions on the existing eight-seed validation cohort."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
WORK=REPO.parent
sys.path.insert(0,str(REPO))
import main_baseline


def main():
    existing=[]
    for root in ('ablation_results','ablation_results_10_6','ablation_results_dvd','ablation_results_11_6'):
        for p in (WORK/root/'sacred').rglob('config.json'):
            c=json.loads(p.read_text(encoding='utf-8'))
            if c.get('learner')=='qtran_learner':existing.append({'path':str(p),'name':c.get('name')})
    runs=[]
    for batch,conditions in enumerate((
        [('3s5z_vs_3s6z',84),('3s5z_vs_3s6z',85),('6h_vs_8z',92),('6h_vs_8z',93)],
        [('3s5z_vs_3s6z',86),('3s5z_vs_3s6z',87),('6h_vs_8z',94),('6h_vs_8z',95)],
    ),1):
        for gpu,(map_name,seed) in enumerate(conditions):
            name='baseline_qtran_base_pymarl2_5m_%s_s%d'%(map_name,seed)
            assert not any(r['name']==name for r in existing),name
            argv=['--debug','--capture=no','--config=qtran','--env-config=sc2','with',
                  'env_args.map_name='+map_name,'seed='+str(seed),'t_max=5050000',
                  'use_tensorboard=True','local_results_path=ablation_results_11_6','name='+name]
            config,params,sources=main_baseline.load_configuration(argv)
            params=main_baseline.prepare_results_path(config,params,WORK)
            effective=main_baseline.inspect_overrides(config,params)
            assert (effective['learner'],effective['mixer'])==('qtran_learner','qtran_base')
            assert (effective['lr'],effective['opt_loss'],effective['nopt_min_loss'])==(.0005,1,.1)
            assert (effective['batch_size_run'],effective['batch_size'],effective['buffer_size'])==(8,128,5000)
            assert effective['test_nepisode']==32 and effective['t_max']==5050000
            assert 'td_lambda' not in effective
            assert not any(k.startswith(('audit_','adaptive_','counterfactual_')) for k in effective)
            runs.append({'batch':batch,'gpu':gpu,'map':map_name,'seed':seed,'name':name,'argv':argv,
                         'command':'CUDA_VISIBLE_DEVICES=%d python3 src/main_baseline.py %s'%(gpu,' '.join(argv)),
                         'effective_config':effective})
    data={'date':'2026-11-10','variant':'QTRAN-base, qtran_paper, small',
          'scope':'Eight existing independent validation seeds, no outcome-based replacement. Original QPLEX cohort reused for comparison.',
          'existing_local_qtran':existing,'runs':runs}
    (HERE/'config_check.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# QTRAN-base外部基线：两批八条固定集合','',
           'QPLEX八颗已齐，下一步QTRAN-base。沿用3s84–87、6h92–95，不按已有表现筛选；第一批3s84/85、6h92/93，第二批3s86/87、6h94/95。每条训练5.05M，每次32局、约10k步测试，全部方向保留。','',
           '## 配方与接入','',
           '已核对[PyMARL2 qtran.yaml](https://github.com/hijkzzz/pymarl2/blob/master/src/config/algs/qtran.yaml)和[learner](https://github.com/hijkzzz/pymarl2/blob/master/src/learners/qtran_learner.py)。本地采用QTRAN-base、qtran_paper、small，8并行环境、batch128、buffer5000、Adam学习率0.0005、epsilon1.0/100k到0.05、Double-Q、200回合硬更新，opt_loss=1、nopt_min_loss=0.1。该learner使用一步TD，不引入BACM的RND/SARSA或QPLEX的TD(lambda)。', '',
           '本地BasicMAC/RNN每步hidden为[B,A,H]；旧代码按上游展平[B*A,H]的布局重排会混淆时间和智能体轴。现在先逐时间步规范成[B,A,H]再stack成[B,T,A,H]，同时兼容两种布局。online和target均修正，loss定义及系数保持。梯度范数改普通数值日志，td_targets日志改为真正目标均值，Q/V日志排除padding。','',
           'main_baseline.py现支持qplex/qtran并分别记录实际源码；QPLEX有效训练配置保持不变，新运行会记录更新后的入口指纹。','',
           '## 同步与预检','',
           '将仓库最新提交同步到服务器src；至少更新main_baseline.py、learners/qtran_learner.py、_test_qtran_baseline.py，预检复用已部署的_test_external_baseline.py。qtran.yaml和modules/mixers/qtran.py需与本仓库一致。','',
           '```bash','cd /home/zhangbei/pymarl2','conda activate pymarl','python3 src/_test_qtran_baseline.py','```','',
           '本地Python3.8/PyTorch2.4.1+cpu共6项通过，含3项旧QPLEX回归，以及QTRAN hidden布局/真实online-target输入核验、连续三次参数更新、loss组合和target同步、配置核验。未在服务器PyTorch1.12.1或SC2/CUDA上运行。', '',
           '四个终端各自进入上述目录并激活环境，分别运行：','']
    for batch in (1,2):
        lines+=['## 第%d批'%batch,'']
        for r in (r for r in runs if r['batch']==batch):
            lines+=['GPU%d：%s seed%d'%(r['gpu'],r['map'],r['seed']),'','```bash',r['command'],'```','']
    lines+=['## 回传与比较','',
            '回传对应sacred和tb_logs。核对实际配置、步数及loss/td_loss/opt_loss/nopt_loss/grad_norm/td_targets，再按最近5M、4–5M、末五次和完整曲线比较。先完成八颗固定集合，不因首批结果换seed或拼接不同配方；QTRAN-alt未实现，不列为本轮方法。','',
            '完成QTRAN-base后再核验独立Qatten，门控拟合诊断和MMM2仍暂缓。外部比较是各算法整套配方比较，不称只替换mixer的结构控制。']
    (HERE/'commands.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'prepared_runs':len(runs),'existing_qtran':len(existing),'first_batch':4},ensure_ascii=False))


if __name__=='__main__':main()
