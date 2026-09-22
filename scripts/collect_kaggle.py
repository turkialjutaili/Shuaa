"""Collect a completed Shuaa Kaggle run; optionally publish its verified ONNX submission.

Authentication stays in local Kaggle OAuth and GitHub CLI credential stores.
This command never opens the test split or changes the final-test lock.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.dataset import file_sha256, load_manifest
from benchmark.evaluation import validate_submission


def run(*args, capture=False):
    return subprocess.run(list(args), cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='turki101/shuaa-training')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/kaggle')
    parser.add_argument('--publish', action='store_true', help='Upload verified release assets, commit the submission and push')
    args = parser.parse_args()
    if not re.fullmatch(r'turki101/[a-z0-9-]+', args.kernel):
        parser.error('Only this project owner\'s Kaggle notebooks are supported')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(args.kernel)
    state = str(status.status).lower()
    print(f'Kaggle status: {state}')
    if not state.endswith('complete'):
        if status.failure_message:
            print(status.failure_message)
        return 0 if state.endswith(('running', 'queued')) else 2
    # Keep resumable .pt checkpoints on Kaggle; collect small scientific records and the final ONNX here.
    pattern = r'(?:^|/)shuaa-output/(?:model\.(?:onnx|parity\.json|submission\.json)|provenance\.json|pip-freeze\.txt|measured-report\.md|runs/campaign\.json|runs/[^/]+/(?:history|result|config)\.json)$'
    api.kernels_output(args.kernel, str(args.output), file_pattern=pattern, quiet=True, page_size=200)
    matches = list(args.output.rglob('model.submission.json'))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one completed model submission; use an output directory dedicated to this run')
    directory = matches[0].parent
    submission = json.loads(matches[0].read_text(encoding='utf-8'))
    manifest = load_manifest(ROOT / 'benchmark/split.json')
    validate_submission(submission, manifest['split_hash'])
    model = directory / 'model.onnx'
    parity = json.loads((directory / 'model.parity.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    if parity['status'] != 'passed' or file_sha256(model) != submission['checkpoint_sha256'] or parity['checkpoint_sha256'] != submission['checkpoint_sha256']:
        raise ValueError('ONNX checksum or exported parity gate failed')
    if provenance['source_commit'] != submission['source_commit'] or provenance['split_hash'] != manifest['split_hash']:
        raise ValueError('Training provenance does not match the submission')
    print(f"Collected {submission['id']}; campaign={provenance['campaign_status']}; test remains untouched")
    if not args.publish:
        print('Pass --publish to create the Release and submit for independent CPU validation.')
        return 0
    if run('gh', 'api', 'user', '--jq', '.login', capture=True).strip() != 'turkialjutaili':
        raise RuntimeError('GitHub must be authenticated as turkialjutaili')
    if run('git', 'remote', 'get-url', 'origin', capture=True).strip() != 'https://github.com/turkialjutaili/Shuaa.git':
        raise RuntimeError('Unexpected Git origin')
    if run('git', 'status', '--porcelain', capture=True).strip():
        raise RuntimeError('Commit or resolve existing local changes before publishing collected results')
    run('git', 'pull', '--ff-only')
    tag = provenance['release_tag']
    if not re.fullmatch(r'training-[0-9]{8}-[0-9]{6}', tag):
        raise ValueError('Unexpected release tag')
    expected_url = f'https://github.com/turkialjutaili/Shuaa/releases/download/{tag}/model.onnx'
    if submission['checkpoint_url'] != expected_url:
        raise ValueError('Release URL differs from training provenance')
    release = subprocess.run(['gh', 'release', 'view', tag, '--repo', 'turkialjutaili/Shuaa'], cwd=ROOT, capture_output=True)
    if release.returncode != 0:
        run('gh', 'release', 'create', tag, str(model), str(directory / 'model.parity.json'), str(directory / 'provenance.json'), str(directory / 'measured-report.md'), '--repo', 'turkialjutaili/Shuaa', '--target', submission['source_commit'], '--title', 'Shuaa trained model ' + tag, '--notes-file', str(directory / 'measured-report.md'))
    else:
        # Never replace an already-published checkpoint under a stable URL.
        from benchmark.evaluation import checkpoint
        checkpoint(submission, ROOT / 'data/checkpoints')
    destination = ROOT / 'submissions' / (submission['id'] + '.json')
    if destination.exists() and destination.read_bytes() != matches[0].read_bytes():
        raise RuntimeError('Submission ID already exists with different bytes')
    shutil.copyfile(matches[0], destination)
    report_dir = ROOT / 'research/runs' / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in ('provenance.json', 'measured-report.md', 'model.parity.json', 'pip-freeze.txt'):
        shutil.copyfile(directory / name, report_dir / name)
    campaign = directory / 'runs/campaign.json'
    if campaign.exists():
        shutil.copyfile(campaign, report_dir / 'campaign.json')
    run('git', 'add', str(destination.relative_to(ROOT)), str(report_dir.relative_to(ROOT)))
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    if changed:
        run('git', 'commit', '-m', 'Submit verified Kaggle-trained Shuaa model and research evidence')
        run('git', 'push', 'origin', 'main')
    print('Submission published. GitHub Actions will compute the validation leaderboard independently.')
    return 0


if __name__ == '__main__':
    # Kaggle writes Unicode notebook logs using the process default encoding.
    # Windows legacy code pages cannot represent the Arabic project name.
    if os.name == 'nt' and not sys.flags.utf8_mode:
        raise SystemExit(subprocess.call([sys.executable, '-X', 'utf8', *sys.argv]))
    sys.exit(main())
