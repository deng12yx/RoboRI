from __future__ import annotations

import asyncio
import copy
import traceback

import numpy as np

from .paths import activate_fs


def serve_fs(paths, method, port):
    activate_fs(paths)
    if method == "fs_mod_ri":
        from .ri_mem.bindings import install_fs_mod_readout
        from .ri_mem.head import load_parameters
        load_parameters(paths["fs_ri_head"])
        install_fs_mod_readout()
    from mme_vla_suite.policies.policy_config import create_trained_policy
    from mme_vla_suite.training.config import get_config
    from challenge_interface.policy import MyPolicy_for_CVPR_Challenge
    from challenge_interface import msgpack_numpy
    import websockets.asyncio.server
    import websockets
    run = paths["cache"] / "fs_checkpoint"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "56000"
    if not checkpoint.exists():
        checkpoint.symlink_to(paths["fs_checkpoint"], target_is_directory=True)
    (run / "history_config.txt").write_text("perceptual-framesamp-modul.yaml")
    base = create_trained_policy(get_config("mme_vla_ftv2_all26_framesamp_modul"), checkpoint, seed=7)
    service = encoder = None
    if method == "fs_mod_ri":
        from .ri_mem.query import QwenService
        service = QwenService(str(paths["qwen_checkpoint"]), paths.get("qwen_device", "cuda:0"), paths["cache"] / "query_plans")
    elif method == "fs_mod_retrieval":
        from .retrieval import Encoder
        encoder = Encoder(paths["retrieval_checkpoint"])

    async def handler(socket):
        policy = copy.copy(base)
        native_transform = base._input_transform
        wrapper = MyPolicy_for_CVPR_Challenge(model=policy)
        wrapper.chunk_size = 20
        packer = msgpack_numpy.Packer()
        engine = bank = None
        await socket.send(packer.pack({"method": method, "chunk_size": 20, "ri_mem": method.endswith("_ri")}))

        def reset():
            nonlocal engine, bank
            if engine is not None:
                engine.close()
            buffer = policy.mem_buffer
            if buffer is not None:
                buffer.__dict__.pop("get_frame_sampling_indices", None)
                buffer.clear()
            policy._input_transform = native_transform
            wrapper.reset()
            bank = engine = None
            if encoder is not None:
                from .retrieval import FrameBank
                bank = FrameBank(encoder)
                bank.bind(policy.mem_buffer)
            if service is not None:
                from .ri_mem.engine import RIMem
                from .ri_mem.bindings import bind_fs_policy
                engine = RIMem(service, paths["fs_ri_head"], capacity=256)
                bind_fs_policy(policy, engine)

        try:
            reset()
            async for payload in socket:
                inputs = msgpack_numpy.unpackb(payload)
                if inputs.get("reset"):
                    reset()
                    result = {"reset_finished": True}
                else:
                    frames = inputs["front_rgb_list"]
                    indices = list(range(policy.step_idx + 1, policy.step_idx + 1 + len(frames)))
                    if bank is not None:
                        bank.add(frames, indices)
                    if engine is not None:
                        if engine.tracker is None:
                            engine.reset(inputs["task_goal"][0], frames[0])
                        for frame, index in zip(frames, indices):
                            engine.observe(frame, index)
                        engine.begin_decision(frames[-1])
                    result = wrapper.infer(inputs)
                    result["actions"] = np.asarray(result["actions"])[:20]
                await socket.send(packer.pack(result))
        except websockets.ConnectionClosed:
            pass
        except Exception as error:
            from .ri_mem.query import InvalidPlan
            if isinstance(error, InvalidPlan):
                await socket.send(packer.pack({"error": "invalid_query_plan", "message": str(error)}))
            else:
                await socket.send(traceback.format_exc())
                raise
        finally:
            if engine is not None:
                engine.close()
            if policy.mem_buffer is not None:
                policy.mem_buffer.__dict__.pop("get_frame_sampling_indices", None)
                policy.mem_buffer.clear()

    async def run_server():
        async with websockets.asyncio.server.serve(handler, "127.0.0.1", port, compression=None,
            max_size=None, ping_interval=None, ping_timeout=None) as server:
            print(f"{method} ready on port {port}", flush=True)
            await server.serve_forever()
    asyncio.run(run_server())
