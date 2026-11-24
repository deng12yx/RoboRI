import numpy as np
import pytest
from robomme_ri.protocol import CONDITIONS, TASKS, EpisodeSpec
from robomme_ri.baselines import METHODS, ABLATIONS
from robomme_ri.perturbations import Perturbation
from robomme_ri.cli import grid
from robomme_ri.evaluate import summarize


class FakeScene:
    def __init__(self):
        self.p = np.array([0.0, 0.0, 0.02])
        self.original = self.p.copy()
        self.held = False
        self.frames = []

    def select_task_object(self, *args):
        return "red_cube"

    def position(self, n):
        return self.p

    def shift_endpoint(self, n, d):
        return self.p + [d, 0, 0]

    def hold(self, n):
        self.held = True

    def release(self, n):
        self.held = False

    def move(self, n, p):
        self.p = np.array(p)

    def role(self, n):
        return "target"

    def required_stage(self, n):
        return 2

    def stage(self):
        return 0

    def can_start(self, *args):
        return True

    def use_imminent(self, n):
        return False

    def endpoint_valid(self, n):
        return True

    def prepare_visual(self, *args):
        pass

    def visual_step(self, c, t, sub):
        self.frames.append(t)

    def restore_visual(self, *args):
        pass


def test_inventory_and_full_grid():
    assert len(TASKS) == 16 and len(CONDITIONS) == 10 and len(ABLATIONS) == 11
    assert sum(m.rimem for m in METHODS.values()) == 3
    specs = list(grid({"methods": ["framesamp-modul"]}))
    assert len(specs) == 8800
    a = EpisodeSpec("framesamp-modul", "PickXtimes", 2, "shift")
    b = EpisodeSpec("simplememvla", "PickXtimes", 2, "shift")
    assert a.event_seed(99) == b.event_seed(99)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_all_ten_control_step_clocks_and_endpoints(condition):
    spec = EpisodeSpec("framesamp-modul", "PickXtimes", 0, condition)
    scene = FakeScene()
    event = Perturbation(spec, 42, scene)
    event.prepare()
    for t in range(110):
        event.before_step(t, t)
        event.after_step(t)
    assert event.result()["event_completed"]
    assert not scene.held
    if condition == "shift":
        np.testing.assert_allclose(scene.p, scene.original + [0.02, 0, 0])
    elif CONDITIONS[condition].group == "remember":
        np.testing.assert_allclose(scene.p, scene.original)
    if condition in ("camera", "light"):
        assert len(scene.frames) == 3  # .15 seconds, strictly < .2 s


def test_late_task_object_event_is_invalidated_and_restored():
    scene = FakeScene()
    event = Perturbation(EpisodeSpec("pi05", "PickXtimes", 0, "mid_arrival"), 42, scene)
    event.prepare()
    event.before_step(0, 0)
    scene.stage = lambda: 2
    event.after_step(1)
    assert event.state == "invalid" and not event.result()["event_completed"]
    np.testing.assert_allclose(scene.p, scene.original)


def test_subtypes_are_balanced_and_invalid_events_are_not_successes():
    for cond in ("unrelated", "target_like", "light"):
        subtypes = [
            EpisodeSpec("pi05", "BinFill", i, cond).resolved_subtype for i in range(50)
        ]
        assert len(set(subtypes)) == 2 and subtypes.count(subtypes[0]) == 25
    rows = [
        dict(
            method="pi05",
            task="BinFill",
            episode=0,
            condition="shift",
            status="success",
            event_completed=False,
        ),
        dict(
            method="pi05",
            task="BinFill",
            episode=1,
            condition="shift",
            status="technical_error",
            event_completed=True,
        ),
        dict(
            method="pi05",
            task="BinFill",
            episode=2,
            condition="shift",
            status="fail",
            event_completed=True,
        ),
    ]
    s = summarize(rows)[0]
    assert (
        s["episodes"] == 1
        and s["macro_tsr"] == 0
        and s["technical_errors"] == 1
        and s["inadmissible_events"] == 1
    )


def test_interface_smoke_and_different_configs_do_not_enter_same_tsr():
    base = dict(
        method="pi05",
        task="BinFill",
        episode=0,
        condition="clean",
        status="success",
        event_completed=True,
    )
    assert summarize([{**base, "evaluation_kind": "interface_smoke"}]) == []
    result = summarize(
        [{**base, "config_hash": "a"}, {**base, "config_hash": "b", "status": "fail"}]
    )
    assert len(result) == 2 and {r["macro_tsr"] for r in result} == {0.0, 100.0}
