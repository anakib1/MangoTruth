import os
from typing import Any

import yaml
from dotenv import load_dotenv

from compute.engine import ComputeEngine
from compute.rabbitmq_broker import RabbitMQBroker
from detectors.mocks import MockDetector


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_config_value(key: str, default: Any) -> Any:
    """Fetch a configuration value from the environment or fallback to a default."""
    return os.getenv(key, default)


if __name__ == "__main__":
    try:
        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "compute_config.yaml")
        config = load_config(config_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(dotenv_path=os.path.join(base_dir, "config", "sample.env"))
        broker_config = config.get("rabbitmq", dict())
        detector_config = config.get("detectors", dict())

        # RabbitMQ Broker Configuration
        print(get_config_value("RABBITMQ_SOURCE_QUEUE_NAME", broker_config.get("source_queue_name")))
        broker = RabbitMQBroker(
            source_queue_name=get_config_value("RABBITMQ_SOURCE_QUEUE_NAME", broker_config.get("source_queue_name")),
            response_queue_name=get_config_value("RABBITMQ_RESPONSE_QUEUE_NAME",
                                                 broker_config.get("response_queue_name")),
            rabbitmq_host=get_config_value("RABBITMQ_HOST", broker_config.get("host")),
            rabbitmq_port=int(get_config_value("RABBITMQ_PORT", broker_config.get("port"))),
            rabbitmq_username=get_config_value("RABBITMQ_USERNAME", broker_config.get("username")),
            rabbitmq_password=get_config_value("RABBITMQ_PASSWORD", broker_config.get("password"))
        )

        # Mock Initialization of Detectors
        detectors = []
        for idx, detector in enumerate(config["detectors"], start=1):
            name = get_config_value(f"DETECTOR_{idx}_NAME", detector["name"])
            labels = get_config_value(f"DETECTOR_{idx}_LABELS", ",".join(detector["labels"])).split(",")
            detectors.append(MockDetector(detector_name=name, labels=labels))
        default_detector = detectors[0]

        # Engine init
        engine = ComputeEngine(detectors=detectors, default_detector=default_detector, broker=broker)
        print("Starting the Compute Engine...")
        engine.start_consuming()

        # Keep the program running
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping the Compute Engine...")
        engine.stop_consuming()
        engine.close()
