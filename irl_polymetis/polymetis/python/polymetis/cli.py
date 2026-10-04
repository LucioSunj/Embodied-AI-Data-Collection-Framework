import atexit
import logging
import os
import signal
import subprocess
import sys
import time

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from polymetis.robot_servers import GripperServerLauncher
from polymetis.utils.data_dir import BUILD_DIR, which
from polymetis.utils.grpc_utils import check_server_exists


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "conf")
)
LEGACY_OVERRIDE_ALIASES = {
    "robot_ip": "robot_client.robot_client.executable_cfg.robot_ip",
}


def _normalize_cfg(cfg):
    """Handle legacy Hydra packaging used by this repository's configs."""
    needs_copy = "_group_" in cfg or (
        cfg.robot_client
        and OmegaConf.is_config(cfg.robot_client)
        and "robot_client" in cfg.robot_client
    )
    if not needs_copy:
        return cfg

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    if "_group_" in cfg and "robot_model" not in cfg:
        cfg.robot_model = cfg._group_
        del cfg["_group_"]

    if (
        cfg.robot_client
        and OmegaConf.is_config(cfg.robot_client)
        and "robot_client" in cfg.robot_client
    ):
        robot_client_group_cfg = cfg.robot_client
        for key, value in robot_client_group_cfg.items():
            if key == "robot_client":
                continue
            cfg[key] = value
        cfg.robot_client = robot_client_group_cfg.robot_client

    return cfg


def _load_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="launch_robot", overrides=overrides or [])


def _normalize_overrides(overrides):
    normalized = []
    for override in overrides:
        if "=" not in override:
            normalized.append(override)
            continue

        key, value = override.split("=", 1)
        normalized_key = LEGACY_OVERRIDE_ALIASES.get(key, key)
        normalized.append(f"{normalized_key}={value}")

    return normalized


def main():
    cfg = _load_cfg(_normalize_overrides(sys.argv[1:]))
    log.info(f"Adding {BUILD_DIR} to $PATH")
    os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]
    cfg = _normalize_cfg(cfg)

    assert not check_server_exists(
        cfg.ip, cfg.port
    ), (
        "Port unavailable; possibly another server found on designated address. "
        "To prevent undefined behavior, start the service on a different port or "
        "kill stale servers with 'pkill -9 run_server'"
    )

    ip = str(cfg.ip)
    port = str(cfg.port)
    gripper_cfg = getattr(cfg, "gripper", None)
    gripper_server = None
    if gripper_cfg:
        gripper_ip = str(gripper_cfg.ip)
        gripper_port = int(gripper_cfg.port)
        assert not check_server_exists(
            gripper_ip, gripper_port
        ), (
            "Gripper port unavailable; possibly another gripper server found on "
            "the designated address. Start the service on a different port or "
            "kill stale servers before retrying."
        )

    log.info("Starting server")
    server_exec_path = which(cfg.server_exec)
    server_cmd = [server_exec_path, "-s", ip, "-p", port]
    if cfg.use_real_time:
        server_cmd.append("-r")

    server_output = subprocess.Popen(
        server_cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setpgrp
    )
    pgid = os.getpgid(server_output.pid)

    if gripper_cfg:
        log.info("Starting gripper server")
        gripper_server = GripperServerLauncher(gripper_ip, gripper_port)
        gripper_server.start()

    def cleanup():
        log.info(f"Killing subprocess with pid {server_output.pid}, pgid {pgid}...")
        if gripper_server is not None:
            log.info("Stopping gripper server...")
            gripper_server.stop()
        os.killpg(pgid, signal.SIGINT)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signal_number, stack_frame: cleanup())

    if cfg.robot_client:
        OmegaConf.resolve(cfg)
        t0 = time.time()
        while not check_server_exists(cfg.ip, cfg.port):
            time.sleep(0.1)
            if time.time() - t0 > cfg.timeout:
                raise ConnectionError("Robot client: Unable to locate server.")
        if gripper_cfg:
            while not check_server_exists(gripper_ip, gripper_port):
                time.sleep(0.1)
                if time.time() - t0 > cfg.timeout:
                    raise ConnectionError("Gripper server: Unable to locate server.")

        log.info("Starting robot client...")
        client = hydra.utils.instantiate(cfg.robot_client, _recursive_=False)
        client.run()
    else:
        signal.pause()
