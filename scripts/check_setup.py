#!/usr/bin/env python3
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='rand100')
    args = parser.parse_args()
    missing = []
    for module in ['torch', 'torchvision', 'habitat', 'habitat_sim', 'habitat_baselines',
                   'ram', 'transformers', 'requests', 'fastdtw', 'quaternion']:
        if importlib.util.find_spec(module) is None:
            missing.append('Python module: ' + module)
    assets = [
        f'data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{args.split}/{args.split}_bertidx.json.gz',
        f'data/datasets/R2R_VLNCE_v1-2_preprocessed/{args.split}/{args.split}_gt.json.gz',
        'checkpoints/gibson-2plus-resnet50.pth',
        'checkpoints/trm_waypoint_predictor.pth',
        'checkpoints/ram_plus_swin_large_14m.pth',
    ]
    for asset in assets:
        if not (ROOT / asset).is_file():
            missing.append(asset)
    dataset = ROOT / assets[0]
    if dataset.is_file():
        with gzip.open(dataset, 'rt') as stream:
            episodes = json.load(stream)['episodes']
        if not episodes:
            missing.append('Dataset contains no episodes')
        scenes = {episode['scene_id'].removeprefix('data/scene_datasets/') for episode in episodes}
        for scene in sorted(scenes):
            if not (ROOT / 'data/scene_datasets' / scene).is_file():
                missing.append('Scene: ' + scene)
        print(f'Dataset: {len(episodes)} episodes, {len(scenes)} scenes')
    if missing:
        print('Setup incomplete:\n' + '\n'.join('  - ' + item for item in missing))
        raise SystemExit(1)
    print('Evaluation assets and importable packages found. Models may still need their Hugging Face / torch.hub cache.')


if __name__ == '__main__':
    main()
