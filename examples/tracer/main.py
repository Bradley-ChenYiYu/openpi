import dataclasses
import logging
import time

import numpy as np
import tyro

from openpi.policies.tracer_policy import make_tracer_example
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    """Command line arguments for the tracer client."""

    # Host and port to connect to the policy server.
    host: str = "localhost"
    port: int | None = 8000

    # Optional API key for servers that require authentication.
    api_key: str | None = None

    # Disable websocket keepalive by default to avoid ping timeout while server is busy
    # with a long model forward pass.
    # ping_interval: float | None = None
    # ping_timeout: float | None = None

    # Number of requests to send to the server.
    num_steps: int = 20

    # Prompt to use for each inference request.
    prompt: str = "do something"


def _make_request(args: Args) -> dict:
    request = make_tracer_example()
    request["prompt"] = args.prompt
    return request


def main(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        # ping_interval=args.ping_interval,
        # ping_timeout=args.ping_timeout,
    )
    logging.info("Server metadata: %s", client.get_server_metadata())

    # Warm up the server so the first timed request is not dominated by model load.
    for _ in range(2):
        client.infer(_make_request(args))

    timings_ms: list[float] = []
    last_action: dict | None = None
    for _ in range(args.num_steps):
        start_time = time.time()
        last_action = client.infer(_make_request(args))
        timings_ms.append(1000.0 * (time.time() - start_time))

        action_shape = np.asarray(last_action["actions"]).shape
        logging.info("Received actions with shape %s", action_shape)
        if "server_timing" in last_action:
            logging.info("Server timing: %s", last_action["server_timing"])

    if timings_ms:
        logging.info(
            "Client inference timing ms: mean=%.2f min=%.2f max=%.2f",
            float(np.mean(timings_ms)),
            float(np.min(timings_ms)),
            float(np.max(timings_ms)),
        )

    if last_action is not None:
        logging.info("Last action keys: %s", sorted(last_action.keys()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))