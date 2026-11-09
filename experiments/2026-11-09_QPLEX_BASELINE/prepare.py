"""Prepare reference-recipe QPLEX commands on the locked validation cohort."""
import json
import sys
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
WORK=REPO.parent
sys.path.insert(0,str(REPO))
import main_baseline


def main():
    historical=[]
    for root in ('ablation_results','ablation_results_10_6','ablation_results_dvd','ablation_results_11_6'):
        for p in (WORK/root/'sacred').rglob('config.json'):
            c=json.loads(p.read_text(encoding='utf-8'))
            if c.get('learner')!='dmaq_qatten_learner':continue
            info=json.loads(p.with_name('info.json').read_text(encoding='utf-8'))
            ts=info.get('test_battle_won_mean_T',[]);v=info.get('test_battle_won_mean',[])
            index=min(range(len(ts)),key=lambda i:abs(ts[i]-5000000)) if ts else None
            historical.append({'config_path':str(p.relative_to(WORK)),'name':c['name'],
                               'recipe':{k:c.get(k) for k in ('lr','batch_size_run','batch_size','buffer_size','td_lambda','epsilon_start','epsilon_anneal_time','mac','agent')},
                               'near_5m_step':ts[index] if index is not None else None,
                               'near_5m_pct':100*v[index] if index is not None else None,
                               'peak_pct':100*max(v) if v else None,
                               'last_loss':info.get('loss',[None])[-1]})
    runs=[]
    for batch,conditions in enumerate((
        [('3s5z_vs_3s6z',84),('3s5z_vs_3s6z',85),('6h_vs_8z',92),('6h_vs_8z',93)],
        [('3s5z_vs_3s6z',86),('3s5z_vs_3s6z',87),('6h_vs_8z',94),('6h_vs_8z',95)],
    ),1):
        for gpu,(map_name,seed) in enumerate(conditions):
            name='baseline_qplex_pymarl2_5m_%s_s%d'%(map_name,seed)
            assert not any(x['name']==name for x in historical),'already exists: '+name
            argv=['--debug','--capture=no','--config=qplex','--env-config=sc2','with',
                  'env_args.map_name='+map_name,'seed='+str(seed),'t_max=5050000','use_tensorboard=True',
                  'local_results_path=ablation_results_11_6','name='+name]
            config,params,sources=main_baseline.load_configuration(argv)
            params=main_baseline.prepare_results_path(config,params,WORK)
            effective=main_baseline.inspect_overrides(config,params)
            assert (effective['batch_size_run'],effective['batch_size'],effective['buffer_size'])==(8,128,5000)
            assert effective['td_lambda']==.6 and effective['test_nepisode']==32
            assert (effective['mac'],effective['agent'])==('basic_mac','rnn')
            assert effective['learner']=='dmaq_qatten_learner' and effective['mixer']=='dmaq'
            assert not any(k.startswith(('audit_','adaptive_','counterfactual_')) for k in effective)
            command='CUDA_VISIBLE_DEVICES=%d python3 src/main_baseline.py %s'%(gpu,' '.join(argv))
            runs.append({'batch':batch,'gpu':gpu,'map':map_name,'seed':seed,'name':name,'argv':argv,
                         'command':command,'effective_config':effective})
    (HERE/'config_check.json').write_text(json.dumps({'date':'2026-11-09','recipe':'Local qplex.yaml, semantically checked against PyMARL2 reference config; common 5.05M budget and SC2 evaluation protocol.',
        'selection':'Entire prior independent validation cohort, no replacement based on results; two batches of four.',
        'runs':runs},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'historical_inventory.json').write_text(json.dumps(historical,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    _,_,sources=main_baseline.load_configuration(['--config=qplex','--env-config=sc2'])
    files=list(sources)+['_test_external_baseline.py']
    with zipfile.ZipFile(HERE/'server_patch.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(REPO/name,'src/'+name)
    with zipfile.ZipFile(HERE/'server_patch.zip') as z:
        assert z.testzip() is None
        for name in files:assert z.read('src/'+name)==(REPO/name).read_bytes()
    lines=['# QPLEX外部基线：两批八条固定集合','',
           '2026-11-09用户将外部算法比较提到当前优先级。门控拟合诊断批次保留准备文件，暂不运行。本轮先QPLEX，随后核验QTRAN-base和独立Qatten再给相应命令；本文件不把dmaq_qatten_learner计为独立Qatten。','',
           '## 配置与比较口径','',
           '使用本地qplex.yaml，逐项核对PyMARL2参考配置：[算法配置](https://github.com/hijkzzz/pymarl2/blob/master/src/config/algs/qplex.yaml)、[默认配置](https://github.com/hijkzzz/pymarl2/blob/master/src/config/default.yaml)、[学习器](https://github.com/hijkzzz/pymarl2/blob/master/src/learners/dmaq_qatten_learner.py)。这是PyMARL2的QPLEX实现，不称原作者代码的逐字复现，也不称完成原论文10M预算评测。','',
           '保留8并行环境、batch128、buffer5000、epsilon从1.0在100k步降至0.05、TD(lambda)=0.6、Adam学习率0.001、double-Q和200回合target硬更新、basic_mac/rnn。参考learner不使用RND。与BACM共同地图、种子、5.05M预算、每次32局且约每10k步评估；训练配方按算法保留，不是只替换mixer的结构控制。','',
           '旧10条QPLEX记录为BM匹配配方/学习率诊断，继续保留并单列；新名字含pymarl2以区别配置，不将两套配方拼接均值。旧3s三颗最近5M为50/62.5/75%，旧6h五颗为0/0/0/0/37.5%；其中部分6h曾学起后回落，完整原始曲线仍应保留。','',
           '新集合锁定3s84–87、6h92–95共八颗，与已有BM、动态BACM、固定0.13独立集合对应。不能只挑学习成功的种子。两批顺序预先锁定，不按首批结果替换或取消其余种子。','',
           '## 同步和预检','',
           '将server_patch.zip的src目录合并到/home/zhangbei/pymarl2/src。包内为独立main_baseline入口、QPLEX实际依赖与测试，不依赖刚才新增的gate_probe代码。QPLEX学习器只做两项兼容处理：one-hot分配跟随目标张量设备、grad_norm记录为普通数值；target定义、loss和优化器保持原实现。','',
           '```bash','cd /home/zhangbei/pymarl2','conda activate pymarl','python3 src/_test_external_baseline.py','```','',
           '本地PyTorch2.4.1+cpu三项检查通过：真实BasicMAC/RNN与QPLEX learner连续三次更新、target同步；穷举二动作三智能体的贪心一致性；配置与来源核验。未在服务器PyTorch1.12.1/SC2/CUDA上运行。','',
           '四个终端分别进入项目目录并激活pymarl后，每批同时运行以下四条。','']
    for batch in (1,2):
        lines+=['## 第%d批'%batch,'']
        for r in (x for x in runs if x['batch']==batch):
            lines+=['GPU%d：%s seed%d'%(r['gpu'],r['map'],r['seed']),'','```bash',r['command'],'```','']
    lines+=['## 回传与后续','',
            '回传本批sacred与tb_logs；先核查真实步数、有限loss/Q/target/梯度和配置，再比较最近5M、4–5M、末五次及完整曲线。学习器grad_norm现为普通数值，避免旧Sacred中序列化成CUDA tensor对象。零胜、回落及正反差值均保留。','',
            '若出现异常报错，--debug --capture=no会保留原始异常；本地预检仅确认合成批次可运行，不预先宣称已解决旧6h训练失稳。后续调配方应另命名并记录预算，保留本轮原始结果。','',
            '补齐八颗QPLEX之后优先QTRAN-base，再独立Qatten；两者需要单独核验入口/实现和配置，本次没有启动或生成它们的正式训练。MMM2继续暂缓。']
    (HERE/'commands.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'locked_runs':len(runs),'batch1':4,'historical_qplex_records':len(historical),'package_files':len(files)},ensure_ascii=False))


if __name__=='__main__':main()
