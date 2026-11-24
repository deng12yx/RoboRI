"""Manage optional model services and one method's evaluation process."""

import argparse
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robomme_ri.cli import load_configuration


def wait_service(host, port, processes, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("Model service exited during initialization")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Model service did not open {host}:{port}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--method", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpu", required=True)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    config = Path(a.config).resolve()
    cfg = load_configuration(config)["policies"][a.method]
    a.output.mkdir(parents=True, exist_ok=True)
    children, logs = [], []

    def interrupted(*args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        commands = cfg.get("server_commands", [])
        if not a.plan_only:
            for i, command in enumerate(commands):
                if not isinstance(command, list) or not all(
                    isinstance(s, str) for s in command
                ):
                    raise ValueError("server_commands must contain argument lists")
                log = (a.output / f"service_{i}.log").open("w")
                logs.append(log)
                args = [s.replace("{gpu}", a.gpu) for s in command]
                children.append(
                    subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                )
            if commands:
                endpoints = [cfg.get("endpoint"), cfg.get("subgoal_endpoint")]
                for endpoint in endpoints:
                    if endpoint:
                        parsed = urlparse(endpoint)
                        wait_service(
                            parsed.hostname,
                            parsed.port,
                            children,
                            cfg.get("service_timeout", 900),
                        )
                if cfg.get("port"):
                    wait_service(
                        cfg.get("host", "127.0.0.1"),
                        cfg["port"],
                        children,
                        cfg.get("service_timeout", 900),
                    )
        cmd = [
            cfg.get("python", sys.executable),
            "-m",
            "robomme_ri.cli",
            "plan" if a.plan_only else "evaluate",
            str(config),
            "--method",
            a.method,
            "--output",
            str(a.output),
        ]
        proc = subprocess.Popen(cmd)
        children.append(proc)
        result = proc.wait()
        if result:
            raise SystemExit(result)
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(children):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
