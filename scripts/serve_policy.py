import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from openpi.policies import tracer_front_left_policy
from openpi.policies import tracer_front_right_policy
from openpi.policies import tracer_policy
from openpi.policies import tracer_side_policy

def _get_warmup_example(train_config: _config.TrainConfig) -> dict | None:
    data = train_config.data
    if isinstance(data, _config.LeRobotTracerFrontLeftDataConfig):
        return tracer_front_left_policy.make_tracer_front_left_example()
    if isinstance(data, _config.LeRobotTracerFrontRightDataConfig):
        return tracer_front_right_policy.make_tracer_front_right_example()
    if isinstance(data, _config.LeRobotTracerSideDataConfig):
        return tracer_side_policy.make_tracer_side_example()
    if isinstance(data, _config.LeRobotTracerDataConfig):
        return tracer_policy.make_tracer_example()
    return None


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Warm up the policy with a dummy input before serving.
    warmup: bool = False
    # Number of warmup inferences to run.
    warmup_steps: int = 1

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    train_config = get_train_config(args)
    match args.policy:
        case Checkpoint():
            checkpoint_dir = args.policy.dir
        case Default():
            checkpoint_dir = DEFAULT_CHECKPOINT[args.env].dir

    return _policy_config.create_trained_policy(
        train_config, checkpoint_dir, default_prompt=args.default_prompt
    )


def get_train_config(args: Args) -> _config.TrainConfig:
    """Resolve the training config for the given args."""
    match args.policy:
        case Checkpoint():
            return _config.get_config(args.policy.config)
        case Default():
            checkpoint = DEFAULT_CHECKPOINT[args.env]
            return _config.get_config(checkpoint.config)


def main(args: Args) -> None:
    train_config = get_train_config(args)
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.warmup:
        example = _get_warmup_example(train_config)
        if example is None:
            logging.warning("Warmup skipped: no example for config %s", train_config.name)
        else:
            logging.info("Warming up policy with dummy input (%s steps)", args.warmup_steps)
            for _ in range(max(args.warmup_steps, 1)):
                try:
                    policy.infer(example)
                except Exception as exc:
                    logging.exception("Warmup failed: %s", exc)
                    break

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
