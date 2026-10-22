from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"

JEV_TASK_DESCRIPTION = (
    "Choose the next navigation action from the typed choices. Use the route "
    "instruction, executed history, previous planning, waypoint geometry, "
    "shared scene context, discriminative RAM object tags, and BLIP visual "
    "captions. Images are not sent. "
    "Continue unfinished instruction steps, backtrack after a deviation, and "
    "stop only when the destination has been reached."
)


def _option_label(option: str) -> str:

    label = str(option).split(".", 1)[0].strip().upper()
    if len(label) != 1 or not label.isalpha():
        raise ValueError("Invalid NavJev action option: {!r}".format(option))
    return label


def _structured_action(
    option: str,
    candidate_index: int,
    angles: Optional[Sequence[float]],
    distances: Optional[Sequence[float]],
    visual_refs: Optional[Sequence[str]],
) -> Dict[str, str]:

    body = str(option).split(".", 1)[1].strip()
    if body.lower() == "stop":
        return {"motion": "stop"}
    if "move back" in body.lower():
        return {"motion": "move back to last position in an opposite direction"}

    motion = re.split(r"\s+to Place\s+\d+", body, maxsplit=1)[0].strip()
    tag_match = re.search(
        r"contains objects such as\s+(.+?)(?:\.|$)", body, flags=re.IGNORECASE
    )
    tags = tag_match.group(1).strip() if tag_match else ""
    action: Dict[str, str] = {"motion": motion}
    if (
        angles is not None
        and distances is not None
        and 0 <= candidate_index < len(angles)
        and candidate_index < len(distances)
    ):
        action["heading"] = "{:.0f}°".format(
            math.degrees(float(angles[candidate_index])) % 360.0
        )
        action["distance"] = "{:.2f} m".format(float(distances[candidate_index]))
    if tags:
        action["ram_tags"] = re.sub(r"\s*\|\s*", ", ", tags)
    return action


def _strip_synthetic_refs(value: str) -> str:

    text = str(value).strip()
    text = re.sub(
        r"\s+to Place\s+\d+\s+which is corresponding to Image\s+\d+,\s*"
        r"and this image contains objects such as\s+",
        "; observed RAM: ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bPlace\s+\d+\b", "navigation node", text)
    text = re.sub(r"\bImage\s+\d+\b", "visual observation", text)
    return text


def _extract_navigation_state(environment_prompt: str) -> Dict[str, str]:

    prompt = str(environment_prompt)
    match = re.search(
        r"Instruction:\s*(.*?)\s*\nHistory:\s*(.*?)\s*\n"
        r"Previous Planning:\s*(.*?)\s*\nAction options \(step \d+\):",
        prompt,
        flags=re.DOTALL,
    )
    if match:
        instruction, history, previous_planning = match.groups()
    else:
        instruction_match = re.match(r"Instruction:\s*(.*)", prompt, re.DOTALL)
        instruction = instruction_match.group(1) if instruction_match else prompt
        history = "The navigation has just begun, with no history."
        previous_planning = "Navigation has just started, with no planning yet."
    return {
        "instruction": _strip_synthetic_refs(instruction),
        "history": _strip_synthetic_refs(history),
        "previous_planning": _strip_synthetic_refs(previous_planning),
    }


def _action_criterion(action: Dict[str, str]) -> str:
    fields = [action["motion"]]
    if "heading" in action:
        fields.append("heading {}".format(action["heading"]))
    if "distance" in action:
        fields.append("distance {}".format(action["distance"]))
    if "visual_caption" in action:
        fields.append("BLIP: {}".format(action["visual_caption"]))
    if "ram_tags" in action:
        fields.append("RAM: {}".format(action["ram_tags"]))
    return "; ".join(fields)


def format_history_action(action: Any) -> str:

    if not isinstance(action, dict):
        return str(action).strip()
    fields = [str(action.get("motion") or "unknown action").strip()]
    if action.get("heading"):
        fields.append("heading {}".format(action["heading"]))
    if action.get("distance"):
        fields.append("distance {}".format(action["distance"]))
    if action.get("visual_caption"):
        fields.append("BLIP: {}".format(action["visual_caption"]))
    if action.get("ram_tags"):
        fields.append("RAM: {}".format(action["ram_tags"]))
    return "; ".join(fields)


def _deduplicate_shared_ram_tags(
    structured_options: Dict[str, Dict[str, str]],
) -> List[str]:

    candidates: List[Tuple[Dict[str, str], List[str]]] = []
    for action in structured_options.values():
        motion = str(action.get("motion", "")).strip().lower()
        if motion == "stop" or motion.startswith("move back"):
            continue
        tags = [
            tag.strip()
            for tag in str(action.get("ram_tags") or "").split(",")
            if tag.strip()
        ]
        if not tags:
            return []
        candidates.append((action, tags))

    if len(candidates) < 2:
        return []

    common = set(tag.casefold() for tag in candidates[0][1])
    for _, tags in candidates[1:]:
        common.intersection_update(tag.casefold() for tag in tags)
    if not common:
        return []

    shared = [tag for tag in candidates[0][1] if tag.casefold() in common]
    for action, tags in candidates:
        remaining = [tag for tag in tags if tag.casefold() not in common]
        if remaining:
            action["ram_tags"] = ", ".join(remaining)
        else:
            action.pop("ram_tags", None)
    return shared


def build_jev_request(
    task_description: str,
    environment_prompt: str,
    action_options: Sequence[str],
    captions: Sequence[str],
    *,
    step: int,
    model: str = DEFAULT_JEV_MODEL,
    angles: Optional[Sequence[float]] = None,
    distances: Optional[Sequence[float]] = None,
    visual_refs: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    caption_offset = 1 if step >= 2 else 0
    structured_options: Dict[str, Dict[str, str]] = {}
    criteria: Dict[str, str] = {}

    for option_index, option in enumerate(action_options):
        label = _option_label(option)
        candidate_index = option_index - caption_offset
        structured_action = _structured_action(
            option, candidate_index, angles, distances, visual_refs
        )
        caption: Optional[str] = None
        if 0 <= candidate_index < len(captions):
            caption = str(captions[candidate_index]).strip() or None

        if caption:
            structured_action["visual_caption"] = caption
        structured_options[label] = structured_action

    shared_scene_tags = _deduplicate_shared_ram_tags(structured_options)
    for option in action_options:
        label = _option_label(option)
        structured_action = structured_options[label]
        candidate_index = list(structured_options).index(label) - caption_offset
        caption = structured_action.get("visual_caption")
        criterion = _action_criterion(structured_action)
        if not caption and "stop" in str(option).lower():
            criterion += " Choose only when the navigation goal has been reached."
        elif captions and candidate_index >= len(captions):
            criterion += " No new candidate image is associated with this action."
        criteria[label] = criterion

    if len(criteria) < 2:
        raise ValueError("Jev requires at least two action options")

    navigation_state = _extract_navigation_state(environment_prompt)
    state = {"instruction": navigation_state["instruction"]}
    if shared_scene_tags:
        state["shared_scene_tags"] = ", ".join(shared_scene_tags)
    state.update(
        {
            "history": navigation_state["history"],
            "previous_planning": navigation_state["previous_planning"],
            "action_options": structured_options,
        }
    )
    payload = {
        "state": state,
        "model": model,
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose exactly one action that best continues the route instruction. "
                    "Use the execution history to avoid repeating completed steps, retain "
                    "the previous plan when still valid, backtrack after a mistake, and stop "
                    "only when the destination is reached. Return only the typed choice."
                ),
                "criteria": criteria,
            },
        },
    }
    return payload, criteria


class CaptionEngine:

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get(
            "NAVJEV_CAPTION_MODEL", DEFAULT_CAPTION_MODEL
        )
        self.prompt = os.environ.get("NAVJEV_CAPTION_PROMPT", "This is a scene of")
        self.max_new_tokens = int(os.environ.get("NAVJEV_CAPTION_MAX_TOKENS", "20"))
        self.device_name = os.environ.get("NAVJEV_CAPTION_DEVICE", "auto")
        self.processor = None
        self.model = None
        self.device = None
        self.dtype = None
        self.family = None
        self.last_unique_count = 0
        self.last_visual_refs: List[str] = []

    def _load(self) -> None:
        if self.model is not None:
            return

        import torch

        self.device = torch.device(
            "cuda"
            if self.device_name == "auto" and torch.cuda.is_available()
            else ("cpu" if self.device_name == "auto" else self.device_name)
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if "blip2" in self.model_name.lower():
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self.family = "blip2"
            self.processor = Blip2Processor.from_pretrained(
                self.model_name, cache_dir="checkpoints/huggingface"
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                cache_dir="checkpoints/huggingface",
            ).to(self.device)
        else:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.family = "blip"
            self.processor = BlipProcessor.from_pretrained(
                self.model_name, cache_dir="checkpoints/huggingface"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                cache_dir="checkpoints/huggingface",
            ).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil_images(rgb_images: Any) -> List[Any]:
        import numpy as np
        import torch
        from PIL import Image

        if not isinstance(rgb_images, torch.Tensor):
            raise TypeError("selected_rgb_images must be a torch.Tensor")
        images = rgb_images.detach().cpu()
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("expected candidate RGB tensor shaped [N,H,W,3]")
        if images.dtype != torch.uint8:
            if images.numel() and float(images.max()) <= 1.0:
                images = images * 255.0
            images = images.clamp(0, 255).to(torch.uint8)
        return [Image.fromarray(np.asarray(image)) for image in images]

    @staticmethod
    def _deduplicate_pil_images(images: Sequence[Any]):

        unique_images: List[Any] = []
        mapping: List[int] = []
        seen: Dict[str, int] = {}
        for image in images:
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest not in seen:
                seen[digest] = len(unique_images)
                unique_images.append(image)
            mapping.append(seen[digest])
        refs = ["V{}".format(index + 1) for index in mapping]
        return unique_images, mapping, refs

    def caption_batch(self, rgb_images: Any) -> List[str]:

        import torch

        self._load()
        images = self._to_pil_images(rgb_images)
        if not images:
            self.last_unique_count = 0
            self.last_visual_refs = []
            return []
        unique_images, mapping, refs = self._deduplicate_pil_images(images)
        self.last_unique_count = len(unique_images)
        self.last_visual_refs = refs

        if self.family == "blip2":
            encoded = self.processor(
                images=unique_images,
                text=[self.prompt] * len(unique_images),
                padding=True,
                return_tensors="pt",
            )
        else:
            encoded = self.processor(
                images=unique_images, padding=True, return_tensors="pt"
            )

        encoded = {
            key: (
                value.to(self.device, dtype=self.dtype)
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        unique_captions = [
            text.strip()
            for text in self.processor.batch_decode(generated, skip_special_tokens=True)
        ]
        captions = [unique_captions[index] for index in mapping]
        return captions


class JevClient:

    def __init__(self) -> None:
        import requests

        self.api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is required to run NavJev"
            )
        self.endpoint = os.environ.get("JEV_ENDPOINT", DEFAULT_JEV_ENDPOINT)
        self.model = os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL)
        self.timeout = float(os.environ.get("JEV_TIMEOUT_SEC", "30"))
        self.max_attempts = int(os.environ.get("JEV_MAX_ATTEMPTS", "3"))
        self.session = requests.Session()
        self.last_payload: Optional[Dict[str, Any]] = None

    def choose(
        self,
        task_description: str,
        environment_prompt: str,
        action_options: Sequence[str],
        captions: Sequence[str],
        *,
        step: int,
        angles: Optional[Sequence[float]] = None,
        distances: Optional[Sequence[float]] = None,
        visual_refs: Optional[Sequence[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        payload, criteria = build_jev_request(
            task_description,
            environment_prompt,
            action_options,
            captions,
            step=step,
            model=self.model,
            angles=angles,
            distances=distances,
            visual_refs=visual_refs,
        )
        self.last_payload = payload
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }

        response = None
        last_error: Optional[Exception] = None
        request_attempts = 0
        retry_statuses: List[int] = []
        retry_after_values: List[str] = []
        for attempt in range(max(1, self.max_attempts)):
            request_attempts = attempt + 1
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code not in (429, 500, 502, 503, 504, 529):
                    break
                retry_statuses.append(int(response.status_code))
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    retry_after_values.append(str(retry_after))
                last_error = RuntimeError(
                    "Jev temporary HTTP status {}".format(response.status_code)
                )
            except Exception as exc:
                last_error = exc
                response = None
            if attempt + 1 < self.max_attempts:
                time.sleep(min(0.75 * (2 ** attempt), 20.0))

        if response is None:
            raise RuntimeError(
                "Jev request did not produce a response: {}".format(last_error)
            )
        response.raise_for_status()
        data = response.json()
        answers = data.get("answers") or {}
        answer = answers.get("next_action") or {}
        choice = str(answer.get("choice", "")).strip().upper()
        if choice not in criteria:
            raise ValueError("Jev selected unknown option {!r}".format(choice))

        usage = data.get("usage") or {}
        metadata = {
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities") or {},
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "request_attempts": request_attempts,
            "retry_statuses": retry_statuses,
            "retry_after_values": retry_after_values,
        }
        return choice, metadata
