from openpi_client import action_chunk_broker
import numpy as np
import pytest

from openpi.policies import aloha_policy
from openpi.policies import tracer_front_left_policy
from openpi.policies import tracer_front_right_policy
from openpi.policies import tracer_side_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_tracer_side_inputs():
    example = tracer_side_policy.make_tracer_side_example()
    inputs = tracer_side_policy.TracerSideInputs(model_type=_config.get_config("pi0_tracer_finetune").model.model_type)(example)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert inputs["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)
    assert tuple(inputs["state"].shape) == (2,)


def test_tracer_front_left_inputs():
    example = tracer_front_left_policy.make_tracer_front_left_example()
    inputs = tracer_front_left_policy.TracerFrontLeftInputs(model_type=_config.get_config("pi0_tracer_finetune").model.model_type)(example)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert np.count_nonzero(inputs["image"]["left_wrist_0_rgb"]) > 0
    assert np.count_nonzero(inputs["image"]["right_wrist_0_rgb"]) == 0


def test_tracer_front_right_inputs():
    example = tracer_front_right_policy.make_tracer_front_right_example()
    inputs = tracer_front_right_policy.TracerFrontRightInputs(model_type=_config.get_config("pi0_tracer_finetune").model.model_type)(example)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert np.count_nonzero(inputs["image"]["left_wrist_0_rgb"]) == 0
    assert np.count_nonzero(inputs["image"]["right_wrist_0_rgb"]) > 0
