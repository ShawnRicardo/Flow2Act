"""CPU-only checks for portable launcher and benchmark bookkeeping."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import tempfile
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from eval_libero.launch_flow2act import family_gpu_assignment, simulator_process, cpu_offload_assignment, continuation_records
from eval_libero.progress_flow2act import SUITES, build_snapshot, write_snapshot


class EvaluationShellTests(unittest.TestCase):
    def run_shell(self, directory, *, overrides=None, arguments=(), returncode=0):
        repo = Path(__file__).resolve().parents[1]
        capture = directory / 'arguments.json'
        fake_python = directory / 'fake policy python'
        fake_python.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["FLOW2ACT_TEST_CAPTURE"]).write_text(json.dumps('
            '{"argv": sys.argv[1:], "cwd": os.getcwd()}))\n'
            'print("fake-policy-stdout", flush=True)\n'
            'print("fake-policy-stderr", file=sys.stderr, flush=True)\n'
            'sys.exit(int(os.environ["FLOW2ACT_TEST_RETURN_CODE"]))\n'
        )
        fake_python.chmod(0o700)
        env = os.environ.copy()
        for name in (
            'POLICY_PYTHON', 'LIBERO_PYTHON', 'PLUS_PYTHON', 'CHECKPOINT',
            'CHECKPOINT_ROOT', 'MODEL_DIR', 'PROCESSOR_PATH', 'LIBERO_ROOT',
            'LIBERO_PLUS_ROOT', 'GPU_IDS', 'BENCHMARKS', 'SUITES', 'SEED',
            'OUTPUT_DIR', 'RESULTS_ROOT', 'CONTINUE_FROM', 'CPU_OFFLOAD_LAYER_MAP',
            'BASE_PORT', 'PROGRESS_INTERVAL', 'RENDER_CHECK_TIMEOUT',
            'SERVER_START_TIMEOUT', 'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX',
        ):
            env.pop(name, None)
        env.update(
            POLICY_PYTHON=str(fake_python),
            CONDA_PREFIX=str(directory / 'envs/flow2act'),
            CHECKPOINT_ROOT=str(directory / 'checkpoints'),
            RESULTS_ROOT=str(directory / 'results'),
            FLOW2ACT_TEST_CAPTURE=str(capture),
            FLOW2ACT_TEST_RETURN_CODE=str(returncode),
        )
        env.update(overrides or {})
        process = subprocess.run(
            ['bash', str(repo / 'scripts/evaluate_flow2act.sh'), *arguments],
            cwd=directory, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=15,
        )
        captured = json.loads(capture.read_text())
        self.assertEqual(captured['cwd'], str(repo))
        return process, captured['argv']

    def test_bare_shell_defaults_and_failure_are_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process, argv = self.run_shell(root, returncode=23)
            value = lambda option: argv[argv.index(option) + 1]
            self.assertEqual(argv[:2], ['-u', 'eval_libero/launch_flow2act.py'])
            self.assertEqual(value('--checkpoint'), str(root / 'checkpoints/Flow2Act/policy_flow2act.ckpt'))
            self.assertEqual(value('--model-dir'), str(root / 'checkpoints/Flow2Act'))
            self.assertEqual(value('--processor-path'), str(root / 'checkpoints/Xiaomi-Robotics-0-Pretrain'))
            self.assertEqual(value('--libero-python'), str(root / 'envs/flow2act-libero/bin/python'))
            self.assertEqual(value('--plus-python'), str(root / 'envs/flow2act-plus/bin/python'))
            self.assertEqual(value('--gpu-ids'), '0,1,2,3,4,5,6,7')
            self.assertEqual(value('--benchmarks'), 'libero,libero_plus')
            self.assertEqual(value('--suites'), 'libero_spatial,libero_object,libero_goal,libero_10')
            output = Path(value('--output-dir'))
            self.assertTrue(output.is_relative_to(root / 'results'))
            log = (output / 'log.log').read_text()
            for message in ('fake-policy-stdout', 'fake-policy-stderr'):
                self.assertIn(message, process.stdout)
                self.assertIn(message, log)
            self.assertEqual(process.returncode, 23)

    def test_checkpoint_override_and_render_argument_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'my run/step=30000.ckpt'
            process, argv = self.run_shell(
                root, overrides={'CHECKPOINT': str(checkpoint)},
                arguments=('--render-check-only',),
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            self.assertEqual(argv[argv.index('--checkpoint') + 1], str(checkpoint))
            self.assertEqual(argv[argv.index('--model-dir') + 1], str(checkpoint.parent))
            self.assertEqual(argv[-1], '--render-check-only')


class EvaluationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = {suite: [{"name": f"{suite}-a", "category": "Camera Viewpoints"},
                                {"name": f"{suite}-b", "category": "Sensor Noise"}] for suite in SUITES}
        self.classification = self.root / 'catalog.json'
        self.classification.write_text(json.dumps(self.catalog))

    def tearDown(self):
        self.temp.cleanup()

    def record(self, *, family='libero_plus', suite='libero_10', status='completed', success=True):
        return dict(benchmark=family, suite=suite, task_id=0, episode_idx=0,
                    task_name=f'{suite}-a', perturbation='Camera Viewpoints', seed=7,
                    status=status, success=success)

    def save(self, row, extra=''):
        path=self.root/row['benchmark']/'shard-0'/row['suite']/'episodes.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row)+'\n'+extra)
        return path

    def snapshot(self, **kw):
        return build_snapshot(self.root, self.classification, benchmarks=('libero_plus',), suites=('libero_10',), **kw)

    def test_family_assignment(self):
        for gpus in [[0], [0,2,6], list(range(8))]:
            self.assertEqual(family_gpu_assignment(gpus, ['libero_plus']), {'libero_plus':gpus})
        self.assertEqual(family_gpu_assignment([0], ['libero','libero_plus']), {'libero':[0]})
        self.assertEqual(family_gpu_assignment(list(range(8)), ['libero','libero_plus']), {'libero':[0,1,2,3], 'libero_plus':[4,5,6,7]})
        with self.assertRaises(ValueError): family_gpu_assignment([0,0], ['libero'])

    def test_optional_offload(self):
        self.assertEqual(cpu_offload_assignment('', [0,2]), {0:0,2:0})
        self.assertEqual(cpu_offload_assignment('2:12', [0,2]), {0:0,2:12})
        with self.assertRaises(ValueError): cpu_offload_assignment('3:12', [0,2])

    def test_libero_only_needs_no_plus_installation(self):
        result=build_snapshot(self.root, self.root/'missing.json', benchmarks=('libero',), suites=('libero_goal',))
        self.assertEqual(result['benchmarks']['libero']['expected'],500)
        self.assertEqual(set(result['benchmarks']), {'libero'})

    def test_plus_subset_denominator_and_policy_failures(self):
        self.save(self.record(success=False))
        result=self.snapshot()['benchmarks']['libero_plus']
        self.assertEqual((result['completed'],result['expected'],result['successes']), (1,2,0))
        self.assertEqual(result['macro_success_rate'],0)
        self.assertEqual(result['perturbations']['Camera Viewpoints']['expected'],1)

    def test_errors_excluded_from_policy_success(self):
        self.save(self.record(status='error',success=None))
        result=self.snapshot()
        self.assertEqual(result['technical_errors'],1)
        self.assertEqual(result['benchmarks']['libero_plus']['completed'],0)
        self.assertFalse(result['complete'])

    def test_partial_line_not_counted(self):
        self.save(self.record(), '{"incomplete":')
        result=self.snapshot()
        self.assertEqual(len(result['ignored_partial_lines']),1)
        self.assertEqual(result['benchmarks']['libero_plus']['completed'],1)
        self.assertFalse(result['complete'])

    def test_duplicates_excluded(self):
        row=self.record();self.save(row,json.dumps(row)+'\n')
        result=self.snapshot()
        self.assertEqual(len(result['duplicate_episodes']),1)
        self.assertEqual(result['benchmarks']['libero_plus']['completed'],0)

    def test_invalid_seed_rejected(self):
        row=self.record();row['seed']=42;self.save(row)
        self.assertEqual(len(self.snapshot()['data_errors']),1)

    def test_progress_and_final_files(self):
        self.save(self.record())
        second=self.record(success=False);second.update(task_id=1,task_name='libero_10-b',perturbation='Sensor Noise')
        path=self.root/'libero_plus/shard-0/libero_10/episodes.jsonl'
        with path.open('a') as f: f.write(json.dumps(second)+'\n')
        result=self.snapshot();self.assertTrue(result['complete'])
        text=write_snapshot(self.root,result,final=True)
        self.assertTrue((self.root/'summary.json').is_file())
        self.assertIn('50.00%',text)

    def test_renderer_uses_explicit_paths_and_physical_gpu(self):
        args=SimpleNamespace(libero_python='/envs/libero/bin/python',plus_python='/envs/plus/bin/python',
                             libero_root=self.root/'my-libero',libero_plus_root=self.root/'my-plus')
        env,prefix=simulator_process(args,{},self.root,'libero_plus',6)
        self.assertEqual(prefix,['/envs/plus/bin/python'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'6')
        self.assertEqual(env['MUJOCO_EGL_DEVICE_ID'],'6')
        self.assertIn(str(args.libero_plus_root),env['PYTHONPATH'])
        self.assertTrue((Path(env['LIBERO_CONFIG_PATH'])/'config.yaml').is_file())

    def test_explicit_continuation_retains_normal_failure_and_source(self):
        from eval_libero import completed_flow2act as completion
        metadata = {"stage": "motion3d", "checkpoint_global_step": 30000}
        row=self.record(success=False)
        row.update(protocol=completion.PLUS_PROTOCOL, policy_metadata=metadata)
        path=self.save(row)
        before=path.read_bytes()
        checkpoint=self.root/'policy.ckpt'
        config=dict(checkpoint=str(checkpoint), seed=7, generation_steps=5,
                    seed_protocol='flow2act', mujoco_versions={'libero_plus':'2.3.7'},
                    render_backend='EGL/headless', vanilla_trials_per_task=50,
                    plus_trials_per_task=1,plus_max_control_steps=720,save_videos=False,
                    suites=['libero_10'],benchmarks=['libero_plus'],policy_metadata=metadata)
        (self.root/'evaluation_config.json').write_text(json.dumps(config))
        args=SimpleNamespace(continue_from=self.root, checkpoint=checkpoint, classification=self.classification,
                             seed=7,benchmarks=['libero_plus'],suites=['libero_10'])
        with patch.object(completion,'CLASSIFICATION_PATH',self.classification):
            snapshot, imported=continuation_records(args,config)
        self.assertEqual(len(imported['libero_plus']),1)
        self.assertFalse(imported['libero_plus'][('libero_10',0,0)]['success'])
        self.assertEqual(path.read_bytes(),before)
        altered={**config,'seed':42}
        with self.assertRaisesRegex(ValueError,'protocol mismatch'):
            continuation_records(args,altered)

    def test_continuation_disabled_is_noop(self):
        self.assertEqual(continuation_records(SimpleNamespace(continue_from=None), {}), (None,{}))

    def mocked_full_launcher(self, cache_environment=None):
        from eval_libero import launch_flow2act as launch
        checkpoint=self.root/'policy.ckpt';checkpoint.touch()
        output=self.root/'mock-launch'
        contract=dict(schema='Flow2Act-XR0-checkpoint-v1',kind='trained_policy',
                      model_config=dict(action_shape=[10,32],num_steps=5),
                      method_config=dict(stage='motion3d',memory_length=1,future_horizon=10,execute_horizon=10))
        metadata=dict(architecture_version=contract['schema'],checkpoint_kind='trained_policy',
                      checkpoint_global_step=30000,stage='motion3d',memory_length=1,
                      future_horizon=10,execute_horizon=10,action_horizon=10)
        fake_contract=ModuleType('mibot.utils.checkpoint_contract_flow2act')
        fake_contract.read_checkpoint_contract_flow2act=lambda *a,**k:contract
        fake_torch=ModuleType('torch');fake_torch.load=lambda *a,**k:dict(global_step=30000)
        commands=[]
        environments=[]
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command);self.pid=1000000+len(commands)
                environments.append(kwargs["env"])
                self.server=Path(command[1]).name == 'server_offload_flow2act.py'
                self.returncode=0
            def poll(self):return None if self.server else 0
            def wait(self,timeout=None):return 0
        args=SimpleNamespace(continue_from=None,output_dir=output,gpu_ids=','.join(map(str,range(8))),
                             benchmarks=['libero','libero_plus'],suites=list(SUITES),cpu_offload_layer_map='',
                             server_python=sys.executable,libero_python=sys.executable,plus_python=sys.executable,
                             libero_root=self.root/'libero',libero_plus_root=self.root/'plus',
                             render_check_only=False,render_backend='egl',checkpoint=checkpoint,model_dir=self.root,
                             seed=7,base_port=28050,render_check_timeout=600,classification=self.classification,
                             processor_path=self.root/'processor',server_start_timeout=1200,progress_interval=30)
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "HF_MODULES_CACHE")}
        environment.update(cache_environment or {})
        with patch.dict(os.environ, environment, clear=True), \
             patch.dict(sys.modules,{'torch':fake_torch,fake_contract.__name__:fake_contract}), \
             patch.object(launch,'check_rendering'),patch.object(launch,'report_progress') as report, \
             patch.object(launch,'metadata',return_value=metadata),patch.object(launch,'log'), \
             patch.object(launch.subprocess,'check_output',return_value='2.3.7'), \
             patch.object(launch.subprocess,'Popen',side_effect=FakeProcess), \
             patch.object(launch.socket,'socket'),patch.object(launch.os,'killpg'):
            launch.run(args)
        self.assertTrue(any(call.kwargs.get('final') for call in report.call_args_list))
        return commands, environments

    def test_mocked_full_launcher_shares_eight_gpus_without_duplicate_shards(self):
        commands, _ = self.mocked_full_launcher()
        servers=[c for c in commands if Path(c[1]).name == 'server_offload_flow2act.py']
        lib=[c for c in commands if Path(c[1]).name == 'worker_flow2act.py']
        plus=[c for c in commands if Path(c[1]).name == 'main_libero_plus_flow2act.py']
        self.assertEqual((len(servers),len(lib),len(plus)),(8,4,8))
        self.assertEqual({c[c.index('--shard-index')+1] for c in plus},set(map(str,range(8))))

    def test_launcher_uses_library_cache_defaults_when_unset(self):
        _, environments = self.mocked_full_launcher()
        for environment in environments:
            for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "HF_MODULES_CACHE"):
                self.assertNotIn(name, environment)

    def test_launcher_preserves_explicit_cache_environment(self):
        explicit = {
            "HF_HOME": "/custom/huggingface",
            "XDG_CACHE_HOME": "/custom/cache",
            "TORCH_HOME": "/custom/torch",
            "HF_MODULES_CACHE": "/custom/modules",
        }
        _, environments = self.mocked_full_launcher(explicit)
        for environment in environments:
            for name, value in explicit.items():
                self.assertEqual(environment[name], value)


if __name__=='__main__':unittest.main()
