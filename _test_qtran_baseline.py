"""QTRAN-base hidden-axis alignment, objective and actual learner checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch as th
from controllers.basic_controller import BasicMAC
import main_baseline
from _test_external_baseline import BaselineTests, Logger

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qtran_verified',ROOT/'learners/qtran_learner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class QTranTests(unittest.TestCase):
    def args(self):
        c,_,_=main_baseline.load_configuration(['--config=qtran','--env-config=sc2'])
        c.update(n_agents=3,n_actions=4,state_shape=6,use_cuda=False,device='cpu')
        return SimpleNamespace(**c)

    def test_hidden_time_and_agent_axes_for_both_controller_layouts(self):
        expected=th.arange(2*5*3*4).reshape(2,5,3,4)
        for flat in (False,True):
            frames=[expected[:,t].reshape(-1,4) if flat else expected[:,t] for t in range(5)]
            self.assertTrue(th.equal(module.stack_agent_hidden_states(frames,2,3),expected))
        old=expected.reshape(2,3,5,-1).transpose(1,2)
        self.assertFalse(th.equal(old,expected))

    def test_real_updates_hidden_inputs_losses_and_target_sync(self):
        th.manual_seed(283);np.random.seed(283)
        args=self.args();batch=BaselineTests().batch(args)
        mac=BasicMAC(batch.scheme,{'agents':3},args)
        logger=Logger();learner=module.QLearner(mac,batch.scheme,logger,args)
        before=[p.detach().clone() for p in learner.params]
        expected={}
        with th.no_grad():
            for key,controller in [('online',learner.mac),('target',learner.target_mac)]:
                controller.init_hidden(batch.batch_size);frames=[]
                for t in range(batch.max_seq_length):
                    controller.forward(batch,t);frames.append(controller.hidden_states.clone())
                expected[key]=th.stack(frames,dim=1)
        observed={}
        original=learner.mixer.forward;target_original=learner.target_mixer.forward
        def online_forward(b,hidden_states,actions=None):
            observed['online']=hidden_states.detach().clone()
            return original(b,hidden_states,actions)
        def target_forward(b,hidden_states,actions=None):
            observed['target']=hidden_states.detach().clone()
            return target_original(b,hidden_states,actions)
        learner.mixer.forward=online_forward;learner.target_mixer.forward=target_forward
        learner.train(batch,0,201)
        self.assertTrue(th.equal(observed['online'],expected['online'][:,:-1]))
        self.assertTrue(th.equal(observed['target'],expected['target'][:,1:]))
        for t in (1,2):learner.train(batch,t*10000,201*(t+1))
        self.assertTrue(any(not th.equal(a,b) for a,b in zip(before,learner.params)))
        self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
        self.assertTrue(all(np.isfinite(float(v)) for v in logger.stats.values()))
        self.assertIsInstance(logger.stats['grad_norm'],float)
        self.assertAlmostEqual(logger.stats['loss'],logger.stats['td_loss']+args.opt_loss*logger.stats['opt_loss']+args.nopt_min_loss*logger.stats['nopt_loss'],places=5)
        for k,v in learner.mac.agent.state_dict().items():self.assertTrue(th.equal(v,learner.target_mac.agent.state_dict()[k]))
        for k,v in learner.mixer.state_dict().items():self.assertTrue(th.equal(v,learner.target_mixer.state_dict()[k]))

    def test_qtran_recipe_and_qplex_still_load(self):
        c,_,s=main_baseline.load_configuration(['--config=qtran','--env-config=sc2'])
        self.assertEqual((c['learner'],c['mixer'],c['lr']),('qtran_learner','qtran_base',.0005))
        self.assertEqual((c['batch_size_run'],c['batch_size'],c['opt_loss'],c['nopt_min_loss']),(8,128,1,.1))
        self.assertIn('learners/qtran_learner.py',s)
        self.assertNotIn('learners/dmaq_qatten_learner.py',s)
        q,_,_=main_baseline.load_configuration(['--config=qplex','--env-config=sc2'])
        self.assertEqual((q['learner'],q['td_lambda']),('dmaq_qatten_learner',.6))


if __name__=='__main__':
    th.set_num_threads(1)
    unittest.main(verbosity=2)
