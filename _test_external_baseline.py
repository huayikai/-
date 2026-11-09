"""Real QPLEX learner/controller smoke test without SC2 or Sacred."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch as th
import main_baseline
from controllers.basic_controller import BasicMAC
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from modules.mixers.dmaq_general import DMAQer

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qplex_cpu_learner',ROOT/'learners/dmaq_qatten_learner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Logger:
    def __init__(self):self.stats={};self.console_logger=self
    def log_stat(self,k,v,t):self.stats[k]=v
    def info(self,*args):pass


class BaselineTests(unittest.TestCase):
    def args(self):
        config,params,_=main_baseline.load_configuration(['--config=qplex','--env-config=sc2'])
        config.update(n_agents=3,n_actions=4,state_shape=6,use_cuda=False,device='cpu')
        return SimpleNamespace(**config)

    def batch(self,args):
        scheme={'state':{'vshape':6},'obs':{'vshape':5,'group':'agents'},
                'actions':{'vshape':(1,),'group':'agents','dtype':th.long},
                'avail_actions':{'vshape':(4,),'group':'agents','dtype':th.int},
                'reward':{'vshape':(1,)},'terminated':{'vshape':(1,),'dtype':th.uint8}}
        batch=EpisodeBatch(scheme,{'agents':3},3,7,
                           preprocess={'actions':('actions_onehot',[OneHot(out_dim=4)])},device='cpu')
        for row,length in enumerate((6,4,2)):
            for t in range(length+1):
                data={'state':np.random.randn(1,1,6).astype('float32'),
                      'obs':np.random.randn(1,1,3,5).astype('float32'),
                      'avail_actions':np.ones((1,1,3,4),dtype='int32'),
                      'actions':np.random.randint(0,4,(1,1,3,1))}
                batch.update(data,bs=slice(row,row+1),ts=t)
                if t<length:
                    batch.update({'reward':np.ones((1,1,1),dtype='float32'),
                                  'terminated':np.array([[[t==length-1]]],dtype='uint8')},
                                 bs=slice(row,row+1),ts=t)
        return batch

    def test_three_real_updates_and_target_sync(self):
        th.manual_seed(172);np.random.seed(172)
        args=self.args();batch=self.batch(args)
        mac=BasicMAC(batch.scheme,{'agents':3},args)
        logger=Logger();learner=module.DMAQ_qattenLearner(mac,batch.scheme,logger,args)
        before=[p.detach().clone() for p in learner.params]
        for t in range(3):learner.train(batch,10000*t,201*(t+1))
        self.assertTrue(any(not th.equal(a,b) for a,b in zip(before,learner.params)))
        self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
        self.assertTrue(all(np.isfinite(float(v)) for v in logger.stats.values()))
        self.assertIsInstance(logger.stats['grad_norm'],float)
        for k,v in learner.mac.agent.state_dict().items():self.assertTrue(th.equal(v,learner.target_mac.agent.state_dict()[k]))
        for k,v in learner.mixer.state_dict().items():self.assertTrue(th.equal(v,learner.target_mixer.state_dict()[k]))

    def test_greedy_joint_action_maximizes_qplex_mixer(self):
        th.manual_seed(173);args=self.args();args.n_actions=2;mixer=DMAQer(args)
        local_q=th.randn(2,3,2);states=th.randn(2,1,6)
        best=local_q.max(-1).values.unsqueeze(1)
        greedy=mixer(best,states,is_v=True)
        for code in range(8):
            actions=th.tensor([(code>>i)&1 for i in range(3)]).expand(2,1,3)
            chosen=local_q.gather(2,actions[:,0].unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            onehot=th.nn.functional.one_hot(actions,2).float()
            value=mixer(chosen,states,is_v=True)+mixer(chosen,states,actions=onehot,max_q_i=best,is_v=False)
            self.assertTrue((value<=greedy+1e-5).all())

    def test_reference_recipe_and_manifest(self):
        config,_,sources=main_baseline.load_configuration(['--config=qplex','--env-config=sc2'])
        self.assertEqual((config['batch_size_run'],config['batch_size'],config['buffer_size']),(8,128,5000))
        self.assertEqual((config['mac'],config['agent'],config['td_lambda']),('basic_mac','rnn',.6))
        self.assertEqual(config['test_nepisode'],32)
        self.assertFalse(any('gate_probe' in p or 'dvd_audit' in p for p in sources))
        self.assertTrue(all((ROOT/p).is_file() for p in sources))


if __name__=='__main__':
    th.set_num_threads(1)
    unittest.main(verbosity=2)
