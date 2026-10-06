"""Output-location checks without Sacred, SMAC or a training process."""
import ast
import datetime
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main_audit
from utils.results import prepare_results_path, resolve_results_path

ROOT = Path(__file__).resolve().parent


class ResultsPathTests(unittest.TestCase):
    def test_fixed_default_uses_project_root(self):
        with tempfile.TemporaryDirectory() as project:
            config = main_audit._read_yaml(ROOT / "config" / "default.yaml")
            self.assertEqual(prepare_results_path(config, ["with"], project), ["with"])
            self.assertEqual(Path(config["local_results_path"]), (Path(project) / "ablation_results_10_6").resolve())
            self.assertEqual(resolve_results_path(config["local_results_path"], project), config["local_results_path"])
            self.assertFalse(Path(config["local_results_path"]).exists())
            fallback = {}
            prepare_results_path(fallback, [], project)
            self.assertEqual(fallback["local_results_path"], config["local_results_path"])

    def test_absolute_cli_path_with_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "实验 outputs" / "fixed")
            params = ["with", "local_results_path=" + json.dumps(target, ensure_ascii=False), "seed=79"]
            config = {"local_results_path": "ignored"}
            rewritten = prepare_results_path(config, params, ROOT.parent)
            expected = str((Path(directory) / "实验 outputs" / "fixed").resolve())
            self.assertEqual(config["local_results_path"], expected)
            self.assertEqual(json.loads(rewritten[1].split("=", 1)[1]), expected)
            self.assertEqual(rewritten[-1], "seed=79")
            self.assertEqual(params[1], "local_results_path=" + json.dumps(target, ensure_ascii=False))

    def test_explicit_relative_path_and_last_override(self):
        with tempfile.TemporaryDirectory() as project:
            config = {}
            params = ["local_results_path=first", "local_results_path='my outputs'"]
            rewritten = prepare_results_path(config, params, project)
            expected = str((Path(project) / "my outputs").resolve())
            self.assertEqual(config["local_results_path"], expected)
            self.assertTrue(all(json.loads(p.split("=", 1)[1]) == expected for p in rewritten))

    def test_invalid_path_fails_before_creating_records(self):
        for value in (None, "", " ", False, 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_results_path(value, ROOT.parent)

    def test_audit_observer_and_sacred_config_share_cli_path(self):
        ex = mock.Mock()
        ex.observers = []
        observer = mock.Mock()
        factory = mock.Mock(return_value=observer)
        entry = SimpleNamespace(ex=ex, logger=mock.Mock(), parse_command=lambda args, key, default: next(
            (a.split("=", 1)[1] for a in args if a.startswith(key + "=")), default))
        modules = {"main": entry, "learners": SimpleNamespace(REGISTRY={}),
                   "learners.dvd_audit_learner": SimpleNamespace(DVDNQLearner=object()),
                   "sacred": SimpleNamespace(),
                   "sacred.observers": SimpleNamespace(FileStorageObserver=SimpleNamespace(create=factory))}
        with tempfile.TemporaryDirectory() as directory:
            target = str((Path(directory) / "a path with spaces").resolve())
            argv = ["--config=dvd_audit_router", "--env-config=sc2", "with", "name=path_test",
                    "env_args.map_name=6h_vs_8z", "local_results_path=" + json.dumps(target)]
            with mock.patch.dict(sys.modules, modules):
                main_audit.main(argv)
            saved = ex.add_config.call_args[0][0]
            self.assertEqual(saved["local_results_path"], target)
            factory.assert_called_once_with(str(Path(target) / "sacred" / "6h_vs_8z" / "path_test"))
            command = ex.run_commandline.call_args[0][0]
            cli_path = next(p.split("=", 1)[1] for p in command if p.startswith("local_results_path="))
            self.assertEqual(json.loads(cli_path), target)
            self.assertIn("utils/results.py", [r["path"] for r in saved["audit_source_manifest"]])

    def test_all_runners_use_the_recorded_path_for_tensorboard(self):
        # Compile the real setup function in isolation, omitting only heavy
        # module imports and run_sequential. This exercises actual logger calls.
        for filename in ("run.py", "per_run.py", "dop_run.py", "on_off_run.py"):
            with self.subTest(runner=filename), tempfile.TemporaryDirectory() as directory:
                tree = ast.parse((ROOT / "run" / filename).read_text(encoding="utf-8"))
                setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
                logger = mock.Mock()
                fake_os = SimpleNamespace(path=os.path, _exit=mock.Mock(), EX_OK=0)
                namespace = {"args_sanity_check": lambda c, log: c, "SN": SimpleNamespace,
                             "Logger": lambda log: logger, "pprint": __import__("pprint"),
                             "datetime": datetime, "os": fake_os,
                             "threading": SimpleNamespace(enumerate=lambda: []),
                             "run_sequential": mock.Mock()}
                exec(compile(ast.Module(body=[setup], type_ignores=[]), filename, "exec"), namespace)
                namespace["run"](mock.Mock(), {"local_results_path": directory, "name": "test",
                                              "use_cuda": False, "use_tensorboard": True}, mock.Mock())
                logged = Path(logger.setup_tb.call_args[0][0])
                self.assertEqual(logged.parent, Path(directory) / "tb_logs")
                self.assertTrue(logged.name.startswith("test__"))
                self.assertEqual(namespace["run_sequential"].call_args.kwargs["args"].local_results_path, directory)

    def test_default_runner_model_save_creates_step_directory(self):
        tree = ast.parse((ROOT / "run" / "run.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_sequential")
        save = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.BoolOp) and isinstance(n.test.values[0], ast.Attribute)
                    and n.test.values[0].attr == "save_model")
        with tempfile.TemporaryDirectory() as directory:
            learner, logger = mock.Mock(), mock.Mock()
            namespace = {"os": os, "args": SimpleNamespace(save_model=True, save_model_interval=100,
                                                           local_results_path=directory, unique_token="test__launch"),
                         "runner": SimpleNamespace(t_env=1234), "model_save_time": 0,
                         "learner": learner, "logger": logger}
            exec(compile(ast.Module(body=[save], type_ignores=[]), "model_save", "exec"), namespace)
            expected = Path(directory) / "models" / "test__launch" / "1234"
            self.assertTrue(expected.is_dir())
            learner.save_models.assert_called_once_with(str(expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
