"""Run a baseline Qwen/LoRA predictor in its native ms-swift environment."""

import argparse
import asyncio
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


async def serve(predictor, native, port):
    from websockets.asyncio.server import serve as websocket_serve
    from openpi_client import msgpack_numpy as codec
    from robomme_ri.protocol import EpisodeSpec

    async def handle(ws):
        await ws.send(
            codec.packb({"native": native, "service": "robomme-ri-subgoals-v1"})
        )
        try:
            async for payload in ws:
                req = codec.unpackb(payload)
                try:
                    if req["op"] == "reset":
                        predictor.reset(
                            EpisodeSpec(**req["spec"]),
                            {"front_rgb_list": req["front"]},
                            req["goal"],
                        )
                        result = {"ok": True}
                    elif req["op"] == "observe":
                        predictor.observe({"front_rgb_list": req["front"]}, req["step"])
                        result = {"ok": True}
                    elif req["op"] == "predict":
                        result = {
                            "subgoal": predictor.predict(req["front"], req["step"])
                        }
                    else:
                        raise ValueError(req["op"])
                    await ws.send(codec.packb(result))
                except Exception:
                    await ws.send(traceback.format_exc())
        finally:
            predictor.close()

    async with websocket_serve(
        handle, "127.0.0.1", port, compression=None, max_size=None, ping_interval=None
    ):
        print(f"Subgoal server ready: {native} on {port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--native",
        choices=(
            "symbolic-simple-subgoal",
            "symbolic-grounded-subgoal",
            "memer",
            "oft-symbolic-simple-subgoal",
            "oft-symbolic-grounded-subgoal",
        ),
        required=True,
    )
    p.add_argument("--adapter", required=True)
    p.add_argument("--qwen-model", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--repo", type=Path, default=HERE.parents[1])
    a = p.parse_args()
    sys.path[:0] = [
        str(a.repo / "robomme_policy_learning/examples/robomme"),
        str(a.repo / "robomme_policy_learning/packages/openpi-client/src"),
    ]
    from robomme_ri.predictors import NativeQwen

    predictor = NativeQwen(
        a.native, {"subgoal_adapter": a.adapter, "qwen_model": a.qwen_model}
    )
    asyncio.run(serve(predictor, a.native, a.port))
