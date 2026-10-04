import ctypes
import os


PYTHON_ROOT_DIR = os.path.dirname(__file__)
TORCH_ISOLATION_BUILD_DIR = os.path.abspath(
    os.path.join(PYTHON_ROOT_DIR, "..", "build", "torch_isolation")
)


def candidate_library_paths(lib_name: str):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        yield os.path.join(conda_prefix, "lib", lib_name)

    yield os.path.join(TORCH_ISOLATION_BUILD_DIR, lib_name)


def preload_conda_cpp_runtime():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    for lib_name in ("libstdc++.so.6", "libgcc_s.so.1"):
        runtime_path = os.path.join(conda_prefix, "lib", lib_name)
        if not os.path.exists(runtime_path):
            continue

        ctypes.CDLL(runtime_path, mode=ctypes.RTLD_GLOBAL)


def load_torch_library(lib_name: str, *, classes: bool = False):
    preload_conda_cpp_runtime()

    import torch

    loader = torch.classes.load_library if classes else torch.ops.load_library
    errors = []
    for lib_path in candidate_library_paths(lib_name):
        try:
            loader(lib_path)
            return
        except OSError as exc:
            errors.append((lib_path, exc))

    first_path, first_error = errors[0]
    last_path, last_error = errors[-1]
    print(
        f"Warning: Failed to load '{lib_name}' from '{first_path}' "
        f"({first_error}). Tried fallback '{last_path}' "
        f"and also failed ({last_error})."
    )
    raise last_error
