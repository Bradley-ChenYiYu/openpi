from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._prefix_tokenizer: _tokenizer.PaligemmaTokenizer | None = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup.
            # Use a dedicated wrapper for prefix-text extraction so `return_prefix_tokens`
            # is treated as a static argument instead of a traced value.
            if getattr(model, "supports_prefix_text_from_hidden", False):
                graphdef, state = nnx.split(model)

                def fun(state: nnx.State, *args: Any, **kwargs: Any):
                    module = nnx.merge(graphdef, state)
                    return model.sample_actions.__func__(module, *args, **kwargs)

                jitted_fn = jax.jit(fun, static_argnames=("return_prefix_tokens",))

                def sample_actions(*args: Any, **kwargs: Any):
                    return jitted_fn(state, *args, **kwargs)

                self._sample_actions = sample_actions
            else:
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _decode_prefix_text(self, token_ids: np.ndarray, token_mask: np.ndarray | None) -> str:
        if token_mask is not None:
            token_ids = token_ids[token_mask.astype(bool)]
        if token_ids.size == 0:
            return ""

        if self._prefix_tokenizer is None:
            self._prefix_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max(1, int(token_ids.shape[0])))

        vocab_size = self._prefix_tokenizer._tokenizer.vocab_size()
        if np.any(token_ids < 0) or np.any(token_ids >= vocab_size):
            raise ValueError("prefix_token_ids contain values outside tokenizer vocabulary range")

        return self._prefix_tokenizer._tokenizer.decode(token_ids.astype(np.int32).tolist())

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if getattr(self._model, "supports_prefix_text_from_hidden", False):
            sample_kwargs.setdefault("return_prefix_tokens", True)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        prefix_token_ids = None
        prefix_token_mask = None
        prefix_text = None
        if isinstance(sample_result, dict):
            actions = sample_result["actions"]
            prefix_token_ids = sample_result.get("prefix_token_ids")
            prefix_token_mask = sample_result.get("prefix_token_mask")
            prefix_text = sample_result.get("prefix_text")
        else:
            actions = sample_result

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            if prefix_token_ids is not None:
                prefix_token_ids = np.asarray(prefix_token_ids[0, ...].detach().cpu())
            if prefix_token_mask is not None:
                prefix_token_mask = np.asarray(prefix_token_mask[0, ...].detach().cpu())
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            if prefix_token_ids is not None:
                prefix_token_ids = np.asarray(prefix_token_ids[0, ...])
            if prefix_token_mask is not None:
                prefix_token_mask = np.asarray(prefix_token_mask[0, ...])

        outputs = self._output_transform(outputs)
        if prefix_text is None and prefix_token_ids is not None:
            prefix_text = self._decode_prefix_text(prefix_token_ids, prefix_token_mask)
        if prefix_text is not None:
            outputs["prefix_text"] = prefix_text
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
