import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tracer_example() -> dict:
    """Creates a random input example for the Tracer policy."""
    return {
        "observation/state": np.random.rand(2),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TracerInputs(transforms.DataTransformFn):
    """Convert tracer inputs into the model's expected input schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Convert image to uint8 HWC for policy inference and training paths.
        base_image = _parse_image(data["observation/image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # FAST models do not mask padded images.
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TracerOutputs(transforms.DataTransformFn):
    """Convert model outputs to tracer action dimensions."""

    def __call__(self, data: dict) -> dict:
        # Tracer uses [linear_x, angular_z] command actions.
        return {"actions": np.asarray(data["actions"][:, :2])}
