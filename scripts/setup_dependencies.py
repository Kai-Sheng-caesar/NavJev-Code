#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, **kwargs):
    return subprocess.run(['git', '-C', str(path), *args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check pinned checkouts without downloading or applying patches')
    args = parser.parse_args()
    dependencies = json.loads((ROOT / 'dependencies.lock.json').read_text())
    for name, spec in dependencies.items():
        target = ROOT / 'third_party' / name
        if not target.exists():
            if args.check:
                raise SystemExit(f'Missing dependency: {target}; run without --check to fetch it')
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', spec['url'], str(target)], check=True)
            if spec['package_root'] != '.':
                git(target, 'sparse-checkout', 'init', '--cone')
                git(target, 'sparse-checkout', 'set', spec['package_root'])
            git(target, 'checkout', '--detach', spec['commit'])
        head = git(target, 'rev-parse', 'HEAD', capture_output=True, text=True).stdout.strip()
        if head != spec['commit']:
            raise SystemExit(f'{target}: expected {spec["commit"]}, got {head}; existing checkout left untouched')
        if spec.get('patch'):
            patch = str(ROOT / spec['patch'])
            applied = subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', patch], capture_output=True)
            if applied.returncode:
                if args.check:
                    raise SystemExit(f'{target}: required compatibility patch is missing')
                git(target, 'apply', '--check', patch)
                git(target, 'apply', patch)
        print(f'{name}: {head}')
    print('Source dependencies ready. Install Python packages separately; see README.md.')


if __name__ == '__main__':
    main()
