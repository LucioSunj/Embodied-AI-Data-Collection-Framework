"""FR3 camera roles shared by the dashboard and inference client.

Serial numbers match /home/lsk/openpi/examples/convert2Lerobot.py CAMERA_IDS
and docs/fr3_two_camera_data.md for the two-object training dataset.
"""

import ast
import json
from pathlib import Path
import subprocess
import sys


CAMERA_SERIALS_BY_ROLE = {
    "left": "348122070707",
    "wrist": "352122270841",
    "right": "347622075736",
}
DEFAULT_CAMERA_ROLES = ("left", "wrist")
TRAINING_CONFIG_PATH = Path("/home/lsk/openpi/src/openpi/training/config.py")
OPENPI_PYTHON = Path("/home/lsk/openpi/.venv/bin/python")


def camera_mapping_from_serials(serials):
    """Map serials to observation keys, independently of enumeration order."""
    serials = list(serials)
    if len(serials) != len(set(serials)):
        raise ValueError("相机序列号不能重复")
    unknown = set(serials) - set(CAMERA_SERIALS_BY_ROLE.values())
    if unknown:
        raise ValueError(f"相机不在训练数据映射中: {', '.join(sorted(unknown))}")
    if CAMERA_SERIALS_BY_ROLE["left"] not in serials:
        raise ValueError("策略输入需要训练时的 left 相机 348122070707")
    return {
        serial: f"observation.images.{role}"
        for role, serial in CAMERA_SERIALS_BY_ROLE.items()
        if serial in serials
    }


def policy_camera_mapping(config_name, config_path=TRAINING_CONFIG_PATH):
    """Parse OpenPI syntax with its own Python, even from Polymetis Python 3.9."""
    try:
        result = subprocess.run(
            [str(OPENPI_PYTHON), "-B", str(Path(__file__).resolve()), config_name, str(config_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("读取 OpenPI 相机配置超时，请检查 OpenPI Python 环境") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "无法读取 OpenPI 训练配置中的相机设置")
    return json.loads(result.stdout)


def _read_policy_camera_mapping(config_name, config_path):
    """Read literal camera_names without importing OpenPI or initializing JAX."""
    tree = ast.parse(Path(config_path).read_text(), filename=str(config_path))
    default_roles = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LeRobotPolymetisDataConfig":
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.target.id == "camera_names":
                    default_roles = ast.literal_eval(field.value)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "TrainConfig":
            continue
        fields = {item.arg: item.value for item in node.keywords}
        name = fields.get("name")
        if not isinstance(name, ast.Constant) or name.value != config_name:
            continue
        data = fields.get("data")
        if not isinstance(data, ast.Call) or not isinstance(data.func, ast.Name) or data.func.id != "LeRobotPolymetisDataConfig":
            raise ValueError(f"{config_name} 不是当前 FR3 关节客户端支持的训练配置")
        roles = default_roles
        for item in data.keywords:
            if item.arg == "camera_names":
                roles = ast.literal_eval(item.value)
        if not isinstance(roles, (list, tuple)) or not roles or not all(isinstance(role, str) for role in roles):
            raise ValueError(f"无法读取 {config_name} 的 camera_names")
        if len(set(roles)) != len(roles) or "left" not in roles or set(roles) - set(CAMERA_SERIALS_BY_ROLE):
            raise ValueError(f"训练配置中的相机角色不受支持: {roles}")
        return {role: CAMERA_SERIALS_BY_ROLE[role] for role in roles}
    raise ValueError(f"未找到训练配置: {config_name}")


if __name__ == "__main__":
    try:
        print(json.dumps(_read_policy_camera_mapping(sys.argv[1], sys.argv[2])))
    except (ValueError, OSError, SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
