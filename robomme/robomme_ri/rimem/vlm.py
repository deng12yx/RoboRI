"""Frozen Qwen3-VL parser/verifier. Separate from baseline Qwen LoRA subgoals."""

import hashlib
import json
from pathlib import Path
from threading import Lock
from .query import PARSE_PROMPT, Plan


class QwenSCQ:
    def __init__(self, model_path, device="cuda:0", cache_dir=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.lock = Lock()

    def generate(self, text, images=()):
        from PIL import Image

        content = [{"type": "image", "image": Image.fromarray(x)} for x in images] + [
            {"type": "text", "text": text}
        ]
        with self.lock, self.torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=384)
            return self.processor.decode(
                ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

    def parse(self, instruction):
        key = hashlib.sha256(instruction.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            raw = path.read_text()
        else:
            raw = self.generate(PARSE_PROMPT + instruction)
        plan = Plan.parse(
            instruction, raw
        )  # Invalid outputs are not silently repaired.
        if path and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw + "\n")
        return plan

    def verify(self, instruction, subgoal, reference, current):
        text = (
            "The first image is the subgoal reference; the second is the current front view. "
            "Check ONLY the specified completion condition. Output JSON with complete (boolean) and confidence (0 to 1). "
            f"Instruction: {instruction}\nSubgoal: {subgoal.description}\nCondition: {subgoal.completion}"
        )
        return json.loads(self.generate(text, (reference, current)))
