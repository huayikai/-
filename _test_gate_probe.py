"""Observer parity and independent interpolation-oracle checks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch as th
import _test_dvd_audit as fixture

ROOT = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('gate_probe_analysis', ROOT/'experiments/2026-11-09_GATE_PROBE/analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class GateProbeTests(unittest.TestCase):
    def test_updates_rng_targets_and_optimizer_unchanged_with_observer(self):
        for detach in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                learners=[];rng=[];logs=[]
                for enabled in (False,True):
                    th.manual_seed(715)
                    args=fixture.args_for()
                    args.seed=85
                    args.name='probe'
                    args.local_results_path=directory
                    args.use_rnd=True
                    args.audit_detach_mixed_hidden=detach
                    args.audit_save_gate_probe=enabled
                    args.audit_gate_probe_steps=[1,4]
                    args.audit_gate_probe_batches=2
                    logger=fixture.FakeLogger()
                    learner=fixture.audit.DVDNQLearner(fixture.FakeMAC(args),{},logger,args)
                    batch=fixture.FakeBatch()
                    for t in range(6):
                        learner.train(batch,t,t)
                    learners.append(learner);rng.append(th.get_rng_state().clone());logs.append(logger.stats)
                self.assertTrue(th.equal(*rng))
                self.assertEqual(logs[0],logs[1])
                for label in ('mac','mixer','target_mac','target_mixer','rnd'):
                    a,b=[getattr(x,label).state_dict() for x in learners]
                    for k in a:self.assertTrue(th.equal(a[k],b[k]),label+'/'+k)
                a,b=[x.optimiser.state_dict() for x in learners]
                for key in a['state']:
                    for k,v in a['state'][key].items():
                        if th.is_tensor(v):self.assertTrue(th.equal(v,b['state'][key][k]))
                        else:self.assertEqual(v,b['state'][key][k])
                files=sorted((Path(directory)/'gate_probes/probe').glob('*.npz'))
                self.assertEqual(len(files),4)
                for p in files:
                    r=analysis.evaluate_file(p)
                    self.assertEqual(r['n'],12)  # excludes padded/post-terminal positions
                    self.assertEqual(r['eligible_count'],12)  # GRU outputs already include current obs
                    for label,stats in r['methods'].items():
                        self.assertGreaterEqual(stats['mse_gap_to_oracle'],-1e-9)

    def test_nonlinear_interior_optimum_and_fallback(self):
        # One transition has Q(g)=ELU(-1+2g), y=0 => exact g=.5.
        # The second is a BM fallback; every admissible candidate stays g=0.
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic.npz'
            pb=np.array([[-1.],[-1.]])
            pd=np.array([[1.],[-1.]])
            g=np.array([.13,0.])
            q=lambda gate:analysis.elu((1-gate[:,None])*pb+gate[:,None]*pd).sum(1)
            np.savez(p,pre_bm=pb,pre_dvd=pd,w2=np.ones((2,1)),b2=np.zeros(2),target=np.zeros(2),
                     gate=g,applied_gate=g,route_target=np.zeros(2),bm_q=q(np.zeros(2)),
                     dvd_q=q(np.ones(2)),mixed_q=q(g),eligible=np.array([True,False]),
                     gate_max=1.,seed=1,t_env=1000000,planned_step=1000000,name='synthetic')
            r=analysis.evaluate_file(p)
            expected=(np.expm1(-1)**2)/2
            self.assertAlmostEqual(r['methods']['oracle_grid_augmented']['mse'],expected)
            self.assertAlmostEqual(r['oracle_gate_mean'],.25)

    def test_off_grid_shuffled_optimum_is_in_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'off_grid.npz'
            pb=np.full((2,1),-1.)
            pd=np.ones((2,1))
            g=np.array([.137,.023])
            q=lambda x:analysis.elu((1-x[:,None])*pb+x[:,None]*pd).sum(1)
            np.savez(p,pre_bm=pb,pre_dvd=pd,w2=np.ones((2,1)),b2=np.zeros(2),target=q(g[::-1]),
                     gate=g,applied_gate=g,route_target=np.zeros(2),bm_q=q(np.zeros(2)),
                     dvd_q=q(np.ones(2)),mixed_q=q(g),eligible=np.ones(2,dtype=bool),
                     gate_max=1.,seed=1,t_env=1000000,planned_step=1000000,name='off_grid')
            r=analysis.evaluate_file(p)
            self.assertAlmostEqual(r['methods']['oracle_grid_augmented']['mse'],0.)
            self.assertGreaterEqual(r['methods']['shuffle']['mse_gap_to_oracle'],0.)


if __name__=='__main__':
    th.set_num_threads(1)
    unittest.main(verbosity=2)
