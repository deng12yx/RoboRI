"""Generate readable multi-baseline experiment configurations."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from robomme_ri.baselines import METHODS, MAIN_METHODS
from robomme_ri.protocol import TASKS, CONDITIONS


def main():
    root = "${LAPSE_ROOT}"
    policies = {}
    port = 8800
    for key, m in METHODS.items():
        base = key.removesuffix("+ri-mem")
        cfg = {}
        if m.backend == "mme":
            native = "symbolic-grounded-subgoal" if key == "memer" else m.native
            ckpt = (
                "pi05_baseline/pi05_baseline/79999"
                if key == "pi05"
                else native + "/79999"
            )
            cfg.update(
                checkpoint="${MODEL_ROOT}/" + ckpt,
                python=root + "/robomme_policy_learning/.venv/bin/python",
            )
            if m.subgoals == "qwen":
                sub = (
                    "memer/grounded_subgoal/checkpoint-1300"
                    if key == "memer"
                    else "qwenvl/"
                    + ("simple_subgoal" if "simple" in key else "grounded_subgoal")
                    + "/checkpoint-1400"
                )
                cfg.update(
                    subgoal_adapter="${MODEL_ROOT}/vlm_subgoal_predictor/" + sub,
                    qwen_model="${QWEN_MODEL}",
                )
        elif m.backend == "simplememvla":
            cfg.update(
                checkpoint="${MODEL_ROOT}/others/simplememvla_robomme",
                python=root + "/.venv-simplememvla/bin/python",
            )
        else:
            port += 1
            cfg["endpoint"] = (
                "http"
                if key.startswith("oft-") or key in ("openvla-oft", "smolvla")
                else "ws"
            ) + f"://127.0.0.1:{port}"
            cfg["python"] = root + "/robomme_policy_learning/.venv/bin/python"
            if key == "hamlet":
                cfg.update(
                    host="127.0.0.1",
                    port=port,
                    python=root + "/HAMLET-Isaac-GR00T/.venv/bin/python",
                )
            if m.subgoals == "qwen":
                kind = "simple_subgoal" if "simple" in key else "grounded_subgoal"
                cfg.update(
                    subgoal_adapter="${MODEL_ROOT}/vlm_subgoal_predictor/qwenvl/"
                    + kind
                    + "/checkpoint-1400",
                    qwen_model="${QWEN_MODEL}",
                )
            if key == "ponderpounce":
                cfg["service_contract"] = "robomme-challenge-v1"
                cfg["official_adapter_required"] = True
        if key == "retrieval-fs-mod":
            cfg["retrieval_encoder"] = "${MODEL_ROOT}/siglip-so400m-patch14-384"
        if key == "smolvla":
            cfg["exec_horizon"] = 50
        elif key in ("memoryvla", "diffusion-policy"):
            cfg["exec_horizon"] = 8
        elif key in ("sam2act", "sam2act-plus"):
            cfg["exec_horizon"] = 1
        if m.rimem:
            cfg.update(
                head="${SUBMIT_ROOT}/checkpoints/" + base + ".npz",
                qwen_model="${QWEN_MODEL}",
                qwen_device="cuda:0",
                plan_cache="${SUBMIT_ROOT}/outputs/scq_plans",
                ablation="full",
            )
        policies[key] = cfg
    common = dict(
        benchmark="robomme",
        model_seed=7,
        max_steps=1300,
        exec_horizon=16,
        save_video=True,
        python_paths=[
            root + "/robomme_policy_learning/src",
            root + "/robomme_policy_learning/packages/openpi-client/src",
            root + "/robomme_policy_learning/examples/robomme",
            root + "/robomme_benchmark/src",
            root + "/SimpleMemVLA",
            root + "/HAMLET-Isaac-GR00T",
        ],
        policies=policies,
    )
    paper = dict(
        common,
        methods=list(MAIN_METHODS),
        tasks=list(TASKS),
        episodes=list(range(50)),
        conditions=["clean", *CONDITIONS],
    )
    selected = [
        "pi05",
        "framesamp-context",
        "framesamp-modul",
        "tokendrop-context",
        "tokendrop-modul",
        "retrieval-fs-mod",
        "simplememvla",
        "memer",
        "ttt-context",
        "simplesg-qwen",
        "groundedsg-qwen",
        "sam2act-plus",
        "memoryvla",
        "hamlet",
        "framesamp-modul+ri-mem",
        "tokendrop-modul+ri-mem",
        "simplememvla+ri-mem",
    ]
    experiment = dict(
        common,
        methods=selected,
        tasks=["BinFill", "PickXtimes", "ButtonUnmaskSwap", "VideoPlaceOrder"],
        episodes=[0, 2, 3],
        conditions=["clean", *CONDITIONS],
    )
    directory = HERE / "configs"
    directory.mkdir(exist_ok=True)
    history_methods = [m for m in METHODS if METHODS[m].history_control] + [
        "framesamp-modul",
        "tokendrop-modul",
        "simplememvla",
        "framesamp-modul+ri-mem",
        "tokendrop-modul+ri-mem",
        "simplememvla+ri-mem",
    ]
    history = dict(
        common,
        methods=history_methods,
        tasks=list(TASKS),
        episodes=list(range(30)),
        conditions=["clean", *CONDITIONS],
    )
    for name, cfg in (
        ("paper", paper),
        ("experiment", experiment),
        ("history", history),
    ):
        (directory / (name + ".json")).write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
