from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Sequence


def _rgb_numpy(batch: Any):
    import numpy as np
    import torch

    if not isinstance(batch, torch.Tensor):
        raise TypeError("expected a torch RGB tensor")
    value = batch.detach().cpu()
    if value.dtype != torch.uint8:
        if value.numel() and float(value.max()) <= 1.0:
            value = value * 255.0
        value = value.clamp(0, 255).to(torch.uint8)
    return np.asarray(value)


def export_debug_step(
    output_root: str,
    *,
    episode_id: str,
    step: int,
    instruction: str,
    global_images: Any,
    candidate_images: Any,
    angles: Sequence[float],
    distances: Sequence[float],
    ram_tags: Sequence[str],
    captions: Sequence[str],
    visual_refs: Sequence[str],
    action_options: Sequence[str],
    jev_payload: Dict[str, Any],
    jev_choice: str,
    jev_metadata: Dict[str, Any],
) -> Dict[str, str]:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    step_dir = Path(output_root).expanduser().resolve() / (
        "episode_{}_step_{:03d}".format(episode_id, step)
    )
    step_dir.mkdir(parents=True, exist_ok=True)

    cardinals = _rgb_numpy(global_images)
    candidates = _rgb_numpy(candidate_images)
    angle_values = list(angles)
    distance_values = list(distances)

    for index, image in enumerate(cardinals):
        Image.fromarray(image).save(step_dir / "cardinal_{:02d}.png".format(index))
    for index, image in enumerate(candidates):
        Image.fromarray(image).save(step_dir / "candidate_{:02d}.png".format(index + 1))

    metadata = {
        "episode_id": str(episode_id),
        "step": int(step),
        "instruction": str(instruction),
        "angle_convention": "counter-clockwise radians; 0 is current heading",
        "candidates": [
            {
                "candidate": index + 1,
                "angle_rad": float(angle_values[index]),
                "angle_deg": float(math.degrees(angle_values[index]) % 360.0),
                "distance_m": float(distance_values[index]),
                "ram_tags": str(ram_tags[index]),
                "blip_caption": str(captions[index]),
                "visual_ref": str(visual_refs[index]),
                "action_option": str(action_options[index]),
                "image": "candidate_{:02d}.png".format(index + 1),
            }
            for index in range(len(candidates))
        ],
        "jev_choice": str(jev_choice),
        "jev_metadata": jev_metadata,
    }
    with open(step_dir / "perception.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    with open(step_dir / "jev_request.json", "w", encoding="utf-8") as stream:
        json.dump(jev_payload, stream, ensure_ascii=False, indent=2)

    count = max(1, len(candidates))
    columns = max(4, count)
    figure = plt.figure(figsize=(4.2 * columns, 12.5), constrained_layout=True)
    grid = figure.add_gridspec(3, columns, height_ratios=[1.0, 1.15, 1.35])

    cardinal_labels = ["0°", "90° clockwise", "180°", "270° clockwise"]
    for index in range(columns):
        axis = figure.add_subplot(grid[0, index])
        axis.axis("off")
        if index < len(cardinals):
            axis.imshow(cardinals[index])
            axis.set_title("Cardinal view: {}".format(cardinal_labels[index]))

    polar = figure.add_subplot(grid[1, :2], projection="polar")
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(1)
    max_distance = max(distance_values) if distance_values else 1.0
    polar.set_ylim(0, max(3.0, math.ceil(max_distance * 2.0) / 2.0))
    polar.set_title("Predicted waypoint directions and distances")
    for index, (angle, distance) in enumerate(zip(angle_values, distance_values)):
        polar.plot([angle, angle], [0, distance], linewidth=2)
        polar.scatter([angle], [distance], s=80)
        polar.annotate(
            "{}\n{:.2f} m".format(chr(65 + index), distance),
            (angle, distance),
            xytext=(5, 5),
            textcoords="offset points",
        )

    info_axis = figure.add_subplot(grid[1, 2:])
    info_axis.axis("off")
    info_axis.text(
        0,
        1,
        "Instruction\n{}\n\nJev choice: {}\nConfidence: {}".format(
            instruction,
            jev_choice,
            jev_metadata.get("confidence"),
        ),
        va="top",
        ha="left",
        wrap=True,
        fontsize=12,
    )

    for index in range(columns):
        axis = figure.add_subplot(grid[2, index])
        axis.axis("off")
        if index >= len(candidates):
            continue
        axis.imshow(candidates[index])
        axis.set_title(
            "Candidate {} · {} · {:.0f}° · {:.2f} m".format(
                chr(65 + index),
                visual_refs[index],
                math.degrees(angle_values[index]) % 360.0,
                distance_values[index],
            ),
            fontsize=11,
        )

    figure.suptitle(
        "NavJev real perception step · episode {} · step {}".format(
            episode_id, step
        ),
        fontsize=16,
    )
    overview_path = step_dir / "overview.png"
    figure.savefig(overview_path, dpi=140)
    plt.close(figure)

    detail_figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    detail_grid = detail_figure.add_gridspec(2, 2)
    for index in range(min(4, len(candidates))):
        cell = detail_grid[index // 2, index % 2].subgridspec(
            1, 2, width_ratios=[1.0, 1.35]
        )
        image_axis = detail_figure.add_subplot(cell[0, 0])
        image_axis.imshow(candidates[index])
        image_axis.axis("off")
        image_axis.set_title(
            "Candidate {} · {} · {:.0f}° · {:.2f} m".format(
                chr(65 + index),
                visual_refs[index],
                math.degrees(angle_values[index]) % 360.0,
                distance_values[index],
            )
        )
        text_axis = detail_figure.add_subplot(cell[0, 1])
        text_axis.axis("off")
        detail_text = (
            "BLIP\n{}\n\nRAM\n{}\n\nAction\n{}".format(
                textwrap.fill(str(captions[index]), width=48),
                textwrap.fill(str(ram_tags[index]).replace(" | ", ", "), width=48),
                textwrap.fill(str(action_options[index]), width=48),
            )
        )
        text_axis.text(0, 1, detail_text, va="top", ha="left", fontsize=11)
    detail_figure.suptitle("Per-candidate BLIP, RAM, and action text", fontsize=16)
    details_path = step_dir / "candidate_details.png"
    detail_figure.savefig(details_path, dpi=140)
    plt.close(detail_figure)

    return {
        "directory": os.fspath(step_dir),
        "overview": os.fspath(overview_path),
        "candidate_details": os.fspath(details_path),
        "perception": os.fspath(step_dir / "perception.json"),
        "jev_request": os.fspath(step_dir / "jev_request.json"),
    }
