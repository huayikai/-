"""CPU checks for optional fixes, with no SMAC/Sacred import or training run."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import torch as th
import torch.nn as nn

from modules.agents.n_rnn_agent import NRNNAgent
from modules.mixers.dvd_counterfactual_router import CounterfactualRouterDVDMixer
from modules.mixers.dvd_counterfactual_router_audit import CounterfactualRouterAuditMixer
from utils.td_lambda_audit import build_td_lambda_targets_audit
import main_audit

ROOT = Path(__file__).resolve().parent


def load_learner(filename, name):
    # Import the file directly: learners/__init__.py eagerly imports SMAC learners.
    spec = importlib.util.spec_from_file_location(name, ROOT / "learners" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_learner("dvd_nq_learner_with_sarsa.py", "_audit_original_learner")
audit = load_learner("dvd_audit_learner.py", "_audit_new_learner")


def args_for(td=True, aux=True, mixer="dvd_counterfactual_router"):
    return SimpleNamespace(
        n_agents=3, n_actions=4, state_shape=6, mixing_embed_dim=8,
        hypernet_embed=8, rnn_hidden_dim=8, dvd_heads=2, gat_embed_dim=4,
        use_orthogonal=False, use_layer_norm=False, abs=False,
        adaptive_bm_abs=False, adaptive_gate_hidden_dim=8,
        adaptive_gate_max=1.0, adaptive_gate_init=0.05, adaptive_norm_eps=1e-6,
        use_cuda=False, optimizer="adam", lr=0.001, use_rnd=False,
        rnd_beta=0.01, gamma=0.99, td_lambda=0.3, grad_norm_clip=10.0,
        learner_log_interval=1, mixer=mixer,
        audit_fix_td_lambda=td, audit_isolate_dvd_aux_hidden=aux,
        takeover_bm_abs=False, takeover_warmup_steps=20_000_000,
        takeover_ramp_steps=4_000_000, takeover_max=1.0,
        takeover_reliability_floor=0.5, takeover_norm_eps=1e-6,
    )


def tensor(values):
    return th.tensor(values, dtype=th.float64).reshape(1, -1, 1)


def explicit_lambda_returns(rewards, done, mask, q_next, gamma, lam):
    """Independent forward-view oracle: finite weighted n-step returns."""
    result = th.zeros_like(rewards)
    for row in range(rewards.size(0)):
        for start in range(rewards.size(1)):
            if not mask[row, start, 0]:
                continue
            end = start
            while end + 1 < rewards.size(1) and not done[row, end, 0] and mask[row, end + 1, 0]:
                end += 1
            length = end - start + 1
            for n in range(1, length + 1):
                n_return = sum(gamma ** k * rewards[row, start + k, 0] for k in range(n))
                if not done[row, start + n - 1, 0]:
                    n_return += gamma ** n * q_next[row, start + n - 1, 0]
                weight = lam ** (n - 1) * (1 - lam) if n < length else lam ** (n - 1)
                result[row, start, 0] += weight * n_return
    return result


class FakeMAC(nn.Module):
    def __init__(self, args):
        super(FakeMAC, self).__init__()
        self.agent = NRNNAgent(5, args)
        self.n_agents = args.n_agents
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().expand(batch_size, self.n_agents, -1)

    def forward(self, batch, t):
        # Actual NMAC constructs contiguous inputs by concatenating features.
        output, self.hidden_states = self.agent(batch["obs"][:, t].contiguous(), self.hidden_states)
        return output


class FakeBatch(dict):
    def __init__(self):
        super(FakeBatch, self).__init__()
        self.batch_size = 3
        self.max_seq_length = 7
        self["obs"] = th.randn(3, 7, 3, 5)
        self["state"] = th.randn(3, 7, 6)
        self["reward"] = th.randn(3, 7, 1)
        self["actions"] = th.randint(0, 4, (3, 7, 3, 1))
        self["filled"] = th.zeros(3, 7, 1)
        self["terminated"] = th.zeros(3, 7, 1)
        for row, length in enumerate((6, 4, 2)):
            self["filled"][row, :length] = 1
            self["terminated"][row, length - 1] = 1


class FakeLogger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, step):
        self.stats[key] = float(value)


class TDTargetTests(unittest.TestCase):
    def test_terminal_reward_is_preserved(self):
        r, done, mask, q = tensor([0, 1]), tensor([0, 1]), tensor([1, 1]), tensor([0, 0])
        actual = build_td_lambda_targets_audit(r, done, mask, q, 3, 0.99, 0.3)
        th.testing.assert_close(actual, tensor([0.297, 1]))
        self.assertTrue(th.equal(original.build_td_lambda_targets(r, done, mask, q, 3, 0.99, 0.3), tensor([0, 0])))

    def test_truncation_bootstraps_last_valid_transition(self):
        r, done, mask, q = tensor([0, 1, 900]), tensor([0, 0, 0]), tensor([1, 1, 0]), tensor([2, 3, 900])
        actual = build_td_lambda_targets_audit(r, done, mask, q, 3, 0.99, 0.3)
        th.testing.assert_close(actual, tensor([0.99 * (0.7 * 2 + 0.3 * 3.97), 3.97, 0]))

    def test_terminal_ignores_nonfinite_next_value(self):
        actual = build_td_lambda_targets_audit(tensor([1]), tensor([1]), tensor([1]), tensor([float('nan')]), 3, 0.99, 0.3)
        th.testing.assert_close(actual, tensor([1]))

    def test_against_forward_view_on_variable_length_batches(self):
        th.manual_seed(100)
        r = th.randn(5, 8, 1, dtype=th.float64)
        q = th.randn_like(r)
        mask, done = th.zeros_like(r), th.zeros_like(r)
        for row, (length, terminal) in enumerate([(8, True), (8, False), (5, True), (3, False), (0, False)]):
            mask[row, :length] = 1
            if terminal:
                done[row, length - 1] = 1
        for gamma in (0.0, 0.99, 1.0):
            for lam in (0.0, 0.3, 1.0):
                with self.subTest(gamma=gamma, td_lambda=lam):
                    actual = build_td_lambda_targets_audit(r, done, mask, q, 3, gamma, lam)
                    expected = explicit_lambda_returns(r, done, mask, q, gamma, lam)
                    th.testing.assert_close(actual, expected)


class AuxiliaryGradientTests(unittest.TestCase):
    def make_forward(self, isolate):
        args = args_for(aux=isolate)
        th.manual_seed(101)
        mixer = CounterfactualRouterAuditMixer(args)
        encoder = nn.GRUCell(5, args.rnn_hidden_dim)
        head = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        inputs = th.randn(6, 5)
        hidden = encoder(inputs, th.zeros(6, args.rnn_hidden_dim)).reshape(2, 1, 3, 8)
        hidden.retain_grad()
        q = head(hidden)[..., 0]
        q.retain_grad()
        state = th.randn(2, 1, 6)
        output = mixer(q, state, hidden)
        return mixer, encoder, head, q, hidden, output, state

    def test_auxiliary_only_isolates_real_agent_parameters(self):
        mixer, encoder, head, q, hidden, output, state = self.make_forward(True)
        mixer.last_counterfactual_dvd_q.square().mean().backward()
        self.assertIsNone(q.grad)
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in mixer.bm_mixer.parameters()))
        self.assertGreater(mixer.gat.W.weight.grad.abs().sum().item(), 0)
        self.assertGreater(mixer.hyper_dvd_w1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in mixer.gate_net.parameters()))

    def test_disabled_flag_reproduces_original_hidden_gradient(self):
        mixer, encoder, head, q, hidden, output, state = self.make_forward(False)
        mixer.last_counterfactual_dvd_q.square().mean().backward()
        self.assertIsNone(q.grad)
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_mixed_loss_keeps_agent_and_expert_paths(self):
        mixer, encoder, head, q, hidden, output, state = self.make_forward(True)
        output.square().mean().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None), 0)
        self.assertGreater(mixer.gat.W.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in mixer.gate_net.parameters()))

    def test_isolation_preserves_values_and_zero_signal_fallback(self):
        args = args_for()
        th.manual_seed(102)
        reference = CounterfactualRouterDVDMixer(args)
        candidate = CounterfactualRouterAuditMixer(args)
        candidate.load_state_dict(reference.state_dict())
        q, state = th.randn(2, 4, 3), th.randn(2, 4, 6)
        for hidden in (th.randn(2, 4, 3, 8), th.zeros(2, 4, 3, 8)):
            expected, actual = reference(q, state, hidden), candidate(q, state, hidden)
            self.assertTrue(th.equal(actual, expected))
            self.assertTrue(th.equal(candidate.last_counterfactual_bm_q, reference.last_counterfactual_bm_q))
            self.assertTrue(th.equal(candidate.last_counterfactual_dvd_q, reference.last_counterfactual_dvd_q))
            self.assertTrue(th.equal(candidate.last_counterfactual_gate, reference.last_counterfactual_gate))
        self.assertTrue(th.equal(actual, candidate.bm_mixer(q, state)))


class LearnerIntegrationTests(unittest.TestCase):
    def test_disabled_fixes_match_original_training_step(self):
        args = args_for(td=False, aux=False)
        args.use_rnd = True
        th.manual_seed(103)
        mac = FakeMAC(args)
        other_mac = copy.deepcopy(mac)
        batch = FakeBatch()
        th.manual_seed(104)
        old = original.DVDNQLearner(mac, {}, FakeLogger(), args)
        th.manual_seed(104)
        new = audit.DVDNQLearner(other_mac, {}, FakeLogger(), args)
        old.train(batch, 0, 0)
        new.train(batch, 0, 0)
        for a, b in zip(old.params, new.params):
            self.assertTrue(th.equal(a, b))
            self.assertEqual(a.grad is None, b.grad is None)
            if a.grad is not None:
                self.assertTrue(th.equal(a.grad, b.grad))
        for key, value in old.logger.stats.items():
            self.assertEqual(value, new.logger.stats[key])
        for a, b in zip(old.target_mac.parameters(), new.target_mac.parameters()):
            self.assertTrue(th.equal(a, b))
        for a, b in zip(old.target_mixer.parameters(), new.target_mixer.parameters()):
            self.assertTrue(th.equal(a, b))
        for a, b in zip(old.rnd.parameters(), new.rnd.parameters()):
            self.assertTrue(th.equal(a, b))
        self.assertEqual(old.rnd_ms.__dict__, new.rnd_ms.__dict__)

    def test_all_variants_and_controls_have_finite_updates(self):
        cases = [(td, aux, 'dvd_counterfactual_router') for td in (False, True) for aux in (False, True)]
        cases += [(td, False, mixer) for td in (False, True) for mixer in ('dvd', 'dvd_takeover')]
        for td, aux, mixer in cases:
            with self.subTest(td=td, aux=aux, mixer=mixer):
                args = args_for(td=td, aux=aux, mixer=mixer)
                th.manual_seed(105)
                mac, batch, logger = FakeMAC(args), FakeBatch(), FakeLogger()
                learner = audit.DVDNQLearner(mac, {}, logger, args)
                before = [p.detach().clone() for p in mac.parameters()]
                learner.train(batch, 0, 0)
                self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, mac.parameters())))
                self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
                self.assertTrue(all(th.isfinite(p.grad).all() for p in learner.params if p.grad is not None))
                self.assertEqual(logger.stats['audit_fix_td_lambda'], int(td))
                self.assertEqual(logger.stats['audit_isolate_dvd_aux_hidden'], int(aux))
                if mixer == 'dvd_takeover':
                    self.assertFalse(learner.mixer.bm_mixer.abs)
                    self.assertEqual(learner.mixer.last_takeover_alpha_mean.item(), 0)


class EntryPointTests(unittest.TestCase):
    def test_configurations_and_override_preview(self):
        for name in main_audit.AUDIT_CONFIGS:
            config, params, sources = main_audit.load_configuration([
                '--config=' + name, '--env-config=sc2', 'with',
                'audit_fix_td_lambda=False', 'env_args.map_name=6h_vs_8z',
            ])
            self.assertEqual(config['learner'], 'dvd_audit_learner')
            self.assertTrue(config['audit_fix_td_lambda'])
            self.assertEqual(len(config['audit_source_manifest']), len(sources))
            for item in config['audit_source_manifest']:
                self.assertEqual(item['sha256'], hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest())
            preview = main_audit.inspect_overrides(copy.deepcopy(config), params)
            self.assertFalse(preview['audit_fix_td_lambda'])
            self.assertEqual(preview['env_args']['map_name'], '6h_vs_8z')

    def test_runtime_registration_and_dispatch(self):
        ex, observer = mock.Mock(), object()
        # Reproduce older Sacred's rejection instead of accepting every file.
        def accept_python_source(filename):
            if not filename.endswith('.py'):
                raise AssertionError('Sacred Source.create expects Python source')
        ex.add_source_file.side_effect = accept_python_source
        registry = {'existing': object()}
        entry = SimpleNamespace(ex=ex, results_path=str(ROOT / '_mock_results'), logger=mock.Mock(),
                                parse_command=lambda args, key, default: next(
                                    (arg.split('=', 1)[1] for arg in args if arg.startswith(key + '=')), default))
        modules = {
            'main': entry, 'learners': SimpleNamespace(REGISTRY=registry),
            'learners.dvd_audit_learner': audit,
            'sacred': SimpleNamespace(),
            'sacred.observers': SimpleNamespace(FileStorageObserver=SimpleNamespace(create=mock.Mock(return_value=observer))),
        }
        ex.observers = []
        argv = ['--config=dvd_audit_router', '--env-config=sc2', 'with',
                'name=audit_entry_test', 'env_args.map_name=6h_vs_8z']
        with mock.patch.dict(sys.modules, modules):
            main_audit.main(argv)
        self.assertIs(registry['dvd_audit_learner'], audit.DVDNQLearner)
        self.assertEqual(len(registry), 2)
        self.assertEqual(ex.observers, [observer])
        self.assertEqual(ex.add_source_file.call_args_list,
                         [mock.call(str(ROOT / name)) for name in main_audit.SOURCE_FILES])
        ex.pre_run_hook.assert_called_once()
        run = mock.Mock()
        ex.pre_run_hook.call_args[0][0](run)
        run.add_resource.assert_has_calls([
            mock.call(str(ROOT / 'config/default.yaml')),
            mock.call(str(ROOT / 'config/envs/sc2.yaml')),
            mock.call(str(ROOT / 'config/algs/dvd_audit_router.yaml')),
        ])
        self.assertEqual(run.add_resource.call_count, 3)
        ex.run_commandline.assert_called_once_with([str(ROOT / 'main_audit.py')] + argv[2:])
        saved_config = ex.add_config.call_args[0][0]
        self.assertTrue(saved_config['audit_fix_td_lambda'])


    def test_real_sacred_records_python_sources_and_yaml_resources(self):
        if importlib.util.find_spec('sacred') is None:
            self.skipTest('Sacred is optional for the CPU checks')
        from sacred import Experiment
        from sacred.observers import FileStorageObserver
        config, _, sources = main_audit.load_configuration([
            '--config=dvd_audit_router', '--env-config=sc2',
        ])
        ex = Experiment('audit_file_records', interactive=True)
        main_audit.register_file_records(ex, sources)
        ex.add_config(config)

        @ex.main
        def recording_only(_run):
            return 'file-recording-only; no environment or training'

        with tempfile.TemporaryDirectory() as directory:
            observer = FileStorageObserver.create(directory)
            ex.observers.append(observer)
            # Host inspection is unrelated to this regression (old Sacred's
            # CPU-info keys differ from newer local py-cpuinfo versions).
            with mock.patch('sacred.initialize.get_host_info', return_value={}):
                run = ex.run(options={'--capture': 'no'})
            self.assertEqual(run.status, 'COMPLETED')
            records = json.loads((Path(observer.dir) / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(len(records['resources']), 3)
            saved_contents = [path.read_bytes() for path in Path(observer.resource_dir).rglob('*') if path.is_file()]
            for name in sources:
                if name.endswith('.yaml'):
                    self.assertIn((ROOT / name).read_bytes(), saved_contents)
            recorded_sources = {str(item[0]).replace('\\', '/') for item in records['experiment']['sources']}
            for name in main_audit.SOURCE_FILES:
                self.assertTrue(any(path.endswith(name) for path in recorded_sources), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
