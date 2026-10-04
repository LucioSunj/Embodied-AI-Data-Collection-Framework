import logging
import os
import sys
import time

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from polymetis.robot_servers import GripperServerLauncher
from polymetis.utils.data_dir import BUILD_DIR
from polymetis.utils.grpc_utils import check_server_exists


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "conf")
)


def _load_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="launch_gripper", overrides=overrides or [])


def _normalize_cfg(cfg):
    # Legacy Hydra group configs are nested under cfg.gripper.gripper.
    if not (
        cfg.gripper
        and OmegaConf.is_config(cfg.gripper)
        and "gripper" in cfg.gripper
    ):
        return cfg

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.gripper = cfg.gripper.gripper
    return cfg


def main():
    cfg = _normalize_cfg(_load_cfg(sys.argv[1:]))
    log.info(f"Adding {BUILD_DIR} to $PATH")
    os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]
    OmegaConf.resolve(cfg)

    assert not check_server_exists(str(cfg.ip), int(cfg.port)), (
        "Gripper port unavailable; possibly another server found on the "
        "designated address. Start the service on a different port or kill "
        "stale servers before retrying."
    )

    if cfg.gripper:
        pid = os.fork()
    else:
        pid = os.getpid()  # doesn't fork so only the server gets launched

    if pid > 0:
        gripper_server = GripperServerLauncher(str(cfg.ip), int(cfg.port))
        gripper_server.run()
        return

    t0 = time.time()
    while not check_server_exists(str(cfg.ip), int(cfg.port)):
        time.sleep(0.1)
        if time.time() - t0 > cfg.timeout:
            raise ConnectionError("Robot client: Unable to locate server.")

    gripper_client = hydra.utils.instantiate(cfg.gripper, _recursive_=False)
    gripper_client.run()
