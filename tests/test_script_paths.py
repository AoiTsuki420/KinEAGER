"""Exercise Bash dispatch with stubbed Python; no models or GPUs are started."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
GIT_BASH = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')
pytestmark = pytest.mark.skipif(not BASH, reason='Bash is required for shell workflow checks')

# Keep real Python/model loading out of these path and dispatch regressions.
DISPATCH = r'''
python() {
  printf '%s\t' "$PWD" "$@" >> "$CALL_LOG"
  printf '\n' >> "$CALL_LOG"
  if [ "${1:-}" = - ]; then cat >/dev/null; fi
  return 0
}
source "$1"
'''


def run_workflow(tmp_path, script, task='kcat', mode='single'):
    checkout = tmp_path / 'checkout with spaces'
    checkout.mkdir()
    for name in ('models/esm/config.json', 'models/molt5/config.json',
                 'weights/main.pt', 'weights/expert.pt', 'weights/index.npy',
                 'pdbrecords.json', 'infer.py', 'infer_ensemble.py',
                 'tools/data/extract_pdb_from_catpred_json.py',
                 'tools/data/extract_lig_sdf_from_smiles.py'):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    # First iteration lacks the IID split; second has one. Both must be visited.
    for name in (f'iteration_1/{task}-seq_test_sequence_40cluster.csv',
                 f'iteration_2/{task}_test_split_0.csv'):
        path = checkout / 'splits' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sequence,reactant_smiles\nMKT,CCO\n', encoding='utf-8')
    if script == 'prepare_structures.sh':
        shutil.rmtree(checkout / 'splits/iteration_1')
    log = checkout / 'calls.log'
    env = dict(os.environ)
    env.update({
        'ROOT': checkout.name, 'DATA_ROOT': 'splits', 'OUT_ROOT': 'results',
        'WORK_ROOT': 'work', 'PDB_JSON': 'pdbrecords.json',
        'PDB_OUT': 'pdbs', 'SDF_OUT': 'sdfs',
        'OURS_WEIGHTS': 'weights/main.pt', 'OURS_SCALER': 'weights/scaler.json',
        'ESM_PATH': 'models/esm', 'MOLT5_PATH': 'models/molt5',
        'EXPERT_CKPT': 'weights/expert.pt', 'TRAIN_EMB_NPY': 'weights/index.npy',
        'OURS_ENV': '', 'MODE': mode, 'ITERS_TO_RUN': '1 2',
        'CALL_LOG': log.as_posix(),
        'HF_HOME': (checkout / 'cache/hf').as_posix(),
        'TRANSFORMERS_CACHE': (checkout / 'cache/hub').as_posix(),
        'TORCH_HOME': (checkout / 'cache/torch').as_posix(),
    })
    result = subprocess.run(
        [BASH, '--noprofile', '--norc', '-c', DISPATCH, 'workflow-test',
         (REPO / 'scripts' / script).as_posix()],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [line.rstrip('\t').split('\t') for line in log.read_text(encoding='utf-8').splitlines()]
    return calls, result.stdout


@pytest.mark.parametrize('script,task,mode,entry', [
    ('infer_kcat.sh', 'kcat', 'single', 'infer.py'),
    ('infer_km.sh', 'km', 'single', 'infer.py'),
    ('infer_kcat.sh', 'kcat', 'moe_softgate', 'infer_ensemble.py'),
])
def test_inference_paths_survive_output_directory_changes(tmp_path, script, task, mode, entry):
    calls, _ = run_workflow(tmp_path, script, task, mode)
    inference = [call for call in calls if len(call) > 1 and call[1].endswith('/' + entry)]
    assert len(inference) == 2
    for call in inference:
        checkout = call[1].rsplit('/', 1)[0]
        assert call[0].startswith(checkout + '/results/')
        flags = ('-csv', '-weights', '-esm', '-molt5') if mode == 'single' else (
            '--test_csv', '--main_ckpt', '--expert_ckpt', '--train_emb_npy', '--esm_path', '--molt5_path')
        for flag in flags:
            assert call[call.index(flag) + 1].startswith(checkout + '/')


def test_structure_preparation_accepts_splits_without_iteration_one(tmp_path):
    calls, stdout = run_workflow(tmp_path, 'prepare_structures.sh')
    tools = [call[1] for call in calls if len(call) > 1 and '/tools/data/' in call[1]]
    assert len(tools) == 2
    assert all('checkout with spaces/tools/data/' in path for path in tools)
    assert '1 csvs were updated' in stdout
