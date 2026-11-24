"""SCQ: supplement §1.1. No simulator subtask/answer is used online."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import re
import numpy as np

PARSE_PROMPT = """Parse this robot instruction. Output ONLY compact JSON with keys e,s.
e maps entity_id to brief description.
s is a list of objects:
{"d":subgoal_text,"t":target_entity_id,"c":observable_completion_condition,"e":evidence_codes}.
Evidence codes: I=object identity, P=prior subgoal completion, M=motion/demonstration history,
S=current spatial state. Combine letters when needed.
Keep all entities, task order, exact counts and final stop. Preserve which target/time in a
demonstration; never invent colors or positions.
Use brief phrases, no indentation. For repeated pickup N times output exactly N 'pick up'
subgoals; final put-down does not add a pickup. Every stage, including watching, needs a
target, meaningful completion and evidence. No null/empty fields. The completion c is a
meaningful condition, never a letter code.
Make repeated stages distinguishable by ordinal/count completion: for two cubes, completions
should be 'one cube in bin' then 'two cubes in bin', not the same condition twice.
For previously demonstrated target locations, retain M in the later placement stage and describe
the target as demonstrated. Watching alone does not resolve current target coordinates.
Example instruction: pick up the red cube then press the button to stop
Example answer: {"e":{"r":"red cube","b":"button"},"s":[{"d":"pick up red cube","t":"r","c":"red cube held by gripper","e":"IS"},{"d":"press button to stop","t":"b","c":"button depressed","e":"S"}]}
Instruction:
"""


class InvalidPlan(ValueError):
    pass


@dataclass(frozen=True)
class Subgoal:
    description: str
    target: str
    completion: str
    evidence: frozenset[str]


@dataclass(frozen=True)
class Plan:
    instruction: str
    entities: dict[str, str]
    subgoals: tuple[Subgoal, ...]

    @classmethod
    def parse(cls, instruction, raw):
        try:
            d = json.loads(raw) if isinstance(raw, str) else raw
            assert set(d) == {"e", "s"} and d["e"] and d["s"]
            assert all(
                isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                for k, v in d["e"].items()
            )
            goals = []
            for s in d["s"]:
                assert set(s) == {"d", "t", "c", "e"}
                assert all(isinstance(v, str) and v.strip() for v in s.values())
                assert s["t"] in d["e"] and set(s["e"]) <= set("IPMS") and set(s["e"])
                assert len(s["c"].split()) >= 2 and s["c"] not in "IPMS"
                goals.append(Subgoal(s["d"], s["t"], s["c"], frozenset(s["e"])))
            # Explicit repetition invariants can be checked without a privileged planner.
            words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
            }
            m = re.search(
                r"(?:pick\s*up|pickup|pick)\b.*?\b(\d+|one|two|three|four|five|six|seven|eight)\s+times",
                instruction.lower(),
            )
            if m:
                n = int(m[1]) if m[1].isdigit() else words[m[1]]
                picks = [
                    s
                    for s in goals
                    if re.search(r"\bpick\s*up\b|\bpickup\b", s.description.lower())
                ]
                assert len(picks) == n and len({s.completion for s in picks}) == n
            if re.search(r"\bstop\b|\bbutton\b", instruction.lower()):
                assert any(
                    "button" in s.description.lower() or "stop" in s.description.lower()
                    for s in goals
                )
            return cls(instruction, dict(d["e"]), tuple(goals))
        except (AssertionError, KeyError, TypeError, ValueError) as e:
            raise InvalidPlan(
                "Invalid SCQ JSON, entity reference, completion, or repetition count"
            ) from e


class QueryTracker:
    """At most one asynchronous visual request, every 4 decisions, max 16.

    Check confidence >= .85 is separate from the heuristic query confidence.
    The latter is 1 for the first 20 decisions of a stage, then .5.
    """

    def __init__(self, plan, verify):
        self.plan, self.verify = plan, verify
        self.stage = self.age = self.decisions = self.requests = 0
        self.pending = None
        self.reference = None
        self.errors = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scq")

    @property
    def active(self):
        return self.plan.subgoals[self.stage]

    @property
    def vector(self):
        return np.array(
            [float(x in self.active.evidence) for x in "SIPM"]
            + [1.0 if self.age < 20 else 0.5, 1.0],
            dtype=np.float32,
        )

    def update(self, current):
        if self.pending is not None and self.pending.done():
            try:
                r = self.pending.result()
                if r.get("complete") is True and float(r.get("confidence", 0)) >= 0.85:
                    if self.stage < len(self.plan.subgoals) - 1:
                        self.stage += 1
                        self.age = 0
                        self.reference = np.array(current, copy=True)
            except Exception as e:
                self.errors.append(f"{type(e).__name__}: {e}")
            self.pending = None
        if self.reference is None:
            self.reference = np.array(current, copy=True)
        # First call returns age=0. Increment only after constructing this query.
        q = self.vector
        if self.decisions % 4 == 0 and self.pending is None and self.requests < 16:
            self.pending = self.pool.submit(
                self.verify,
                self.plan.instruction,
                self.active,
                self.reference.copy(),
                np.array(current, copy=True),
            )
            self.requests += 1
        self.decisions += 1
        self.age += 1
        return q

    def close(self):
        if self.pending is not None:
            self.pending.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
