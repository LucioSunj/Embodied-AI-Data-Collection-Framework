from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_GENERATED_PB2 = Path(__file__).resolve().parents[2] / "build" / "polymetis_pb2.py"
_SPEC = spec_from_file_location("_polymetis_generated_pb2", _GENERATED_PB2)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load generated protobuf module from {_GENERATED_PB2}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in _MODULE.__dict__.items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value

