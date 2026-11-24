from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


PARSE_PROMPT = '''Parse this robot instruction. Output ONLY compact JSON with keys e,s.
e maps entity_id to brief description.
s is a list of objects: {"d":subgoal_text,"t":target_entity_id,"c":observable_completion_condition,"e":evidence_codes}.
Evidence codes: I=object identity, P=prior subgoal completion, M=motion/demonstration history,
S=current spatial state. Combine letters when needed.
Keep all entities, task order, exact counts and final stop. Preserve which target/time in a
demonstration; never invent colors or positions. Use brief phrases, no indentation.
For repeated pickup N times output exactly N pick-up subgoals; final put-down does not add a pickup.
Every stage, including watching, needs a target, meaningful completion and evidence. No null/empty fields.
Make repeated stages distinguishable by ordinal/count completion, e.g. one cube in bin, then two cubes in bin.
For previously demonstrated locations retain M in the later placement stage. Watching alone does not
resolve current target coordinates.
Instruction:
'''


class InvalidPlan(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    entities: dict
    subgoals: tuple[dict, ...]

    @classmethod
    def parse(cls, raw, instruction=None):
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
            entities, stages = obj["e"], obj["s"]
            assert isinstance(entities, dict) and entities and isinstance(stages, list) and stages
            assert all(isinstance(v, str) and v.strip() for v in entities.values())
            for stage in stages:
                assert set(stage) == {"d", "t", "c", "e"}
                assert all(isinstance(v, str) and v.strip() for v in stage.values())
                assert stage["t"] in entities and set(stage["e"]) <= set("IPMS")
                assert len(stage["c"].split()) >= 2 and stage["c"] not in ("done", "completed")
            assert len({s["c"] for s in stages}) == len(stages), "Repeated completions must be distinguishable"
            if instruction:
                if re.search(r"\bthen\b", instruction, re.I):
                    mentioned = sorted((instruction.lower().find(text.lower()), key) for key, text in entities.items()
                                       if text.lower() in instruction.lower())
                    expected = [key for _, key in mentioned]
                    actual = list(dict.fromkeys(stage["t"] for stage in stages if stage["t"] in expected))
                    assert actual == expected, "Entity order conflicts with the instruction"
                verbs = list(re.finditer(r"\b(?:pick(?:\s+up)?|lift|pour|place|put|open|close|press|stop)\b", instruction.lower()))
                for action, aliases in (("pour", r"pour"), ("pick", r"pick(?:\s+up)?|lift")):
                    count, explicit = 0, False
                    for i, verb in enumerate(verbs):
                        if not re.fullmatch(aliases, verb[0]):
                            continue
                        clause = instruction.lower()[verb.end():verbs[i + 1].start() if i + 1 < len(verbs) else len(instruction)]
                        match = re.search(r"\b(twice|thrice|two times|three times|\d+ times)\b", clause)
                        if match:
                            word = match[1]
                            amount = {"twice": 2, "thrice": 3, "two times": 2, "three times": 3}.get(word)
                            count += amount if amount is not None else int(word.split()[0])
                            explicit = True
                        else:
                            count += 1
                    if explicit:
                        actual = sum(bool(re.search(rf"\b(?:{aliases})\b", stage["d"].lower())) for stage in stages)
                        assert actual == count, f"Expected exactly {count} {action} subgoals, received {actual}"
        except (KeyError, TypeError, ValueError, AssertionError) as error:
            raise InvalidPlan(str(error) or "Invalid structured subgoal plan") from error
        return cls(entities, tuple(stages))


class QueryTracker:
    def __init__(self, plan: Plan, verifier, reference, instruction):
        self.plan, self.verifier, self.reference, self.instruction = plan, verifier, reference, instruction
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ri-verifier")
        self.pending = None
        self.stage = self.entered = self.requests = 0
        self.trace = []

    @property
    def active(self):
        return self.plan.subgoals[min(self.stage, len(self.plan.subgoals) - 1)]

    def update(self, decision, image):
        if self.pending is not None and self.pending.done():
            try:
                response = self.pending.result()
                confirmed = response["complete"] is True and float(response["confidence"]) >= 0.85
                self.trace.append({"decision": decision, "stage": self.stage, "response": response})
                if confirmed and self.stage + 1 < len(self.plan.subgoals):
                    self.stage += 1
                    self.entered = decision
            finally:
                self.pending = None
        if self.pending is None and decision % 4 == 0 and self.requests < 16:
            self.pending = self.pool.submit(self.verifier, self.instruction, dict(self.active),
                                            self.reference.copy(), image.copy())
            self.requests += 1

    def vector(self, decision):
        codes = self.active["e"]
        confidence = 1.0 if decision - self.entered < 20 else 0.5
        return np.array([float(c in codes) for c in "SIPM"] + [confidence, 1], dtype=np.float32)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class QwenService:
    def __init__(self, checkpoint, device="cuda:0", cache=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from pathlib import Path
        from threading import Lock
        self.torch, self.lock = torch, Lock()
        self.checkpoint = str(checkpoint)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.cache = Path(cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, images=(), max_tokens=384):
        from PIL import Image
        content = [{"type": "image", "image": Image.fromarray(np.asarray(im, np.uint8))} for im in images]
        content.append({"type": "text", "text": prompt})
        with self.lock, self.torch.inference_mode():
            inputs = self.processor.apply_chat_template([{ "role": "user", "content": content}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
            raw = self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())

    def plan(self, instruction):
        import hashlib
        key = hashlib.sha256((PARSE_PROMPT + self.checkpoint + instruction).encode()).hexdigest()
        path = self.cache / f"{key}.json" if self.cache else None
        if path and path.exists():
            return Plan.parse(json.loads(path.read_text()), instruction)
        try:
            raw = self.generate(PARSE_PROMPT + instruction)
        except json.JSONDecodeError as error:
            raise InvalidPlan("Instruction parser returned invalid JSON") from error
        plan = Plan.parse(raw, instruction)
        if path:
            path.write_text(json.dumps(raw, ensure_ascii=False))
        return plan

    def verify(self, instruction, subgoal, reference, image):
        return self.generate(f"Task: {instruction}\nSubgoal: {subgoal['d']}\nCompletion: {subgoal['c']}\n"
            'Compare reference and current images. Return ONLY JSON {"complete":true/false,"confidence":0.0-1.0}.',
            [reference, image], max_tokens=64)

    def detect(self, image, entities, frame):
        from .evidence import Region
        response = self.generate('Identify visible task entities and clearly task-irrelevant distractors. '
            f'Entity names: {json.dumps(entities)}. Return ONLY JSON {{"regions":[{{"label":"entity description '
            'or distractor description","box":[x0,y0,x1,y1],"confidence":0.0,"distractor":false}]}}. '
            'Use normalized 0–1 image coordinates. Do not infer hidden objects. Use exact entity descriptions.', [image])
        return [Region(frame, r["label"], tuple(r["box"]), float(r["confidence"]), bool(r["distractor"]))
                for r in response["regions"]]
