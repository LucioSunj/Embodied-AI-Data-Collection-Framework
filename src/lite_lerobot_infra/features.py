from __future__ import annotations

from typing import Any

import numpy as np

from .types import FeatureSpec

ACTION = "action"
OBSERVATION = "observation"

DEFAULT_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _validate_feature_names(features: dict[str, dict]) -> None:
    invalid = [name for name in features if "/" in name]
    if invalid:
        raise ValueError(f"Feature names should not contain '/': {invalid}")


def hw_to_dataset_features(
    hw_features: FeatureSpec,
    prefix: str,
    *,
    use_videos: bool = True,
) -> dict[str, dict]:
    features: dict[str, dict] = {}
    numeric_keys = [key for key, value in hw_features.items() if value is float]
    image_keys = {key: value for key, value in hw_features.items() if isinstance(value, tuple)}

    if numeric_keys and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(numeric_keys),),
            "names": list(numeric_keys),
        }

    if numeric_keys and prefix == OBSERVATION:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(numeric_keys),),
            "names": list(numeric_keys),
        }

    for key, shape in image_keys.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def combine_feature_dicts(*feature_dicts: dict[str, dict]) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for feature_dict in feature_dicts:
        combined.update(feature_dict)
    return combined


def build_dataset_frame(
    dataset_features: dict[str, dict],
    values: dict[str, Any],
    prefix: str,
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    for key, feature in dataset_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        if feature["dtype"] == "float32" and len(feature["shape"]) == 1:
            frame[key] = np.asarray([values[name] for name in feature["names"]], dtype=np.float32)
        elif feature["dtype"] in {"image", "video"}:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame
